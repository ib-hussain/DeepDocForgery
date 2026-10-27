from __future__ import annotations

import argparse
import copy
import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import deepdocforgery.train as train
from deepdocforgery.decoder import SynergyDenoisingDecoder
from deepdocforgery.io import load_yaml
from deepdocforgery.spatial import ADNSupervision, ArtifactDecouplingLoss
from deepdocforgery.telemetry import RunLogger


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("value", [-24.0, 24.0])
def test_adn_direct_half_input_cannot_overflow_the_zero_term(value):
    # Direct FP16 is deliberately tested without CUDA autocast. CUDA's sum
    # policy differs; this is a portability guard, not proof of the old NaN.
    logits = torch.full((11, 1, 16, 16), value, dtype=torch.float16, requires_grad=True)
    assert torch.isfinite(logits).all()
    assert not torch.isfinite(logits.sum() * 0)
    output = SimpleNamespace(
        adn_auxiliary_logits=logits,
        adn_features={"s32": torch.ones(11, 4, 16, 16, dtype=torch.float16)},
    )
    losses = ArtifactDecouplingLoss()(output, ADNSupervision(torch.zeros_like(logits)))
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["nontext_bce"] == 0
    assert losses["text_decoupling"] == 0
    assert losses["domain_alignment"] == 0
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_weighted_pool_half_forward_and_gradients_match_fp32():
    features = torch.full((1, 2, 64, 64), 20.0, dtype=torch.float16, requires_grad=True)
    weights = torch.ones(1, 1, 64, 64, dtype=torch.float16, requires_grad=True)
    assert not torch.isfinite((features * weights).sum((-2, -1))).all()
    pooled = SynergyDenoisingDecoder._weighted_pool(features, weights)
    torch.testing.assert_close(pooled, torch.full((1, 2), 20.0))
    pooled.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(weights.grad).all()


class ToyModel(torch.nn.Module):
    def __init__(self, *, large=False, dropout=0.0):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 500.0 if large else 0.25)
        self.dropout = torch.nn.Dropout(dropout)
        self.front_end = SimpleNamespace(fusion=SimpleNamespace(level_names=("s8",)))
        self.dropout_values = []
        self.saved_refs = []

    def forward(self, rgb, **kwargs):
        del kwargs
        dropped = self.dropout(rgb)
        self.dropout_values.append(dropped.detach().cpu().clone())
        value = self.linear(dropped)
        saved = value.square()
        self.saved_refs.append(weakref.ref(saved))
        return SimpleNamespace(
            value=saved,
            front_end=SimpleNamespace(fusion=SimpleNamespace(
                mean_attention=lambda: {"s8": {"spatial": saved.mean().sigmoid()}}
            )),
        )


def toy_batch(device="cpu"):
    return SimpleNamespace(
        rgb=torch.full((4, 2), 100.0, device=device), metadata=None, exact_dct=None,
        supervision=SimpleNamespace(image_valid=torch.ones(4, 1, device=device)),
    )


@pytest.mark.parametrize("reject", [False, True])
def test_no_batch_graph_survives_helper_return(reject):
    model = ToyModel()
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def criterion(output, _):
        total = output.value.mean() * (float("nan") if reject else 1.0)
        return {"total": total, "component": total}

    for _ in range(3):
        values, _, bad = train._forward_backward_batch(
            model, criterion, toy_batch(), device=torch.device("cpu"),
            use_amp=False, scaler=scaler, accumulation_steps=4,
        )
        gc.collect()
        assert all(ref() is None for ref in model.saved_refs)
        assert bool(bad) == reject
        assert all(isinstance(value, float) for value in values.values())
    assert (model.linear.weight.grad is None) == reject


@pytest.mark.parametrize("enabled", [False, True])
def test_bf16_retry_replays_rng_and_keeps_previous_accumulation(enabled):
    torch.manual_seed(17)
    model = ToyModel(large=True, dropout=0.5)
    model.linear.weight.grad = torch.full_like(model.linear.weight, 3.0)
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state()
    scaler = torch.amp.GradScaler("cpu", enabled=enabled, init_scale=32.0)
    def criterion(output, _):
        return {"total": output.value.mean()}
    values, _, bad, first = train._batch_with_precision_recovery(
        model, criterion, toy_batch(), device=torch.device("cpu"), use_amp=True,
        scaler=scaler, accumulation_steps=4, allow_bf16_retry=True,
    )
    assert first is not None and not np.isfinite(first["total"])
    assert not bad and np.isfinite(values["total"])
    assert len(model.dropout_values) == 2
    torch.testing.assert_close(model.dropout_values[0], model.dropout_values[1])
    after_retry_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    reference.linear.weight.grad = torch.full_like(reference.linear.weight, 3.0)
    train._forward_backward_batch(
        reference, criterion, toy_batch(), device=torch.device("cpu"), use_amp=True,
        scaler=scaler, accumulation_steps=4, amp_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(model.linear.weight.grad, reference.linear.weight.grad)
    assert torch.equal(after_retry_rng, torch.get_rng_state())
    gc.collect()
    assert all(ref() is None for ref in model.saved_refs)


def test_failed_retry_leaves_existing_gradients_unchanged():
    model = ToyModel()
    model.linear.weight.grad = torch.full_like(model.linear.weight, 7.0)
    def criterion(output, _):
        return {"total": output.value.mean() * float("nan")}
    _, _, bad, first = train._batch_with_precision_recovery(
        model, criterion, toy_batch(), device=torch.device("cpu"), use_amp=True,
        scaler=torch.amp.GradScaler("cpu", enabled=False), accumulation_steps=4,
        allow_bf16_retry=True,
    )
    assert bad == ["total"] and first is not None
    torch.testing.assert_close(model.linear.weight.grad, torch.full_like(model.linear.weight, 7.0))
    gc.collect()
    assert all(ref() is None for ref in model.saved_refs)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1e30])
def test_bad_norm_never_changes_weights_or_adam_moments(enabled, bad_value):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=enabled, init_scale=32.0)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    train._optimizer_step(model, optimizer, scaler, 5.0)
    original = model.weight.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    model.weight.grad = torch.full_like(model.weight, bad_value)
    scale = scaler.get_scale()
    norm, updated = train._optimizer_step(model, optimizer, scaler, 5.0)
    assert not updated and not np.isfinite(norm)
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    for key, value in state["state"][0].items():
        torch.testing.assert_close(optimizer.state_dict()["state"][0][key], value, rtol=0, atol=0)
    assert model.weight.grad is None
    assert scaler.get_scale() == (scale / 2 if enabled else scale)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    _, updated = train._optimizer_step(model, optimizer, scaler, 5.0)
    assert updated


def fixture_config(tmp_path, epochs=2):
    records = []
    for split, count in (("train", 5), ("val", 2)):
        for i in range(count):
            image_path = tmp_path / f"{split}-{i}.png"
            mask_path = tmp_path / f"{split}-{i}-mask.png"
            rng = np.random.default_rng(i)
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image_path)
            mask = np.zeros((64, 64), dtype=np.uint8)
            if i % 2:
                mask[16:48, 16:48] = 255
            Image.fromarray(mask).save(mask_path)
            records.append(dict(sample_id=f"{split}-{i}", image=str(image_path),
                                mask=str(mask_path), split=split, label=i % 2,
                                adn_supervision="proxy"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = load_yaml("configs/cpu/smoke.yaml")
    config["data"] = dict(manifest=str(manifest), batch_size=1, num_workers=0,
                          image_size=[64, 64], pin_memory=False)
    config["training"].update(epochs=epochs, cpu_threads=2, amp=False,
                              gradient_accumulation_steps=4, save_every_epochs=1,
                              output_dir=str(tmp_path / "run"), selection_metric="pixel/f1")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    args = argparse.Namespace(config=path, manifest=None, device="cpu", output=None,
                              resume="auto", fresh=False, epochs=None, log_dir=tmp_path / "logs",
                              no_progress=True, maximum_train_batches=None, maximum_val_batches=None)
    return args


def test_training_flushes_after_last_skip_and_resumes_unchanged_config(tmp_path, monkeypatch):
    args = fixture_config(tmp_path)
    original_config = args.config.read_bytes()
    original_batch = train._batch_with_precision_recovery
    calls = 0

    def injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls in (2, 5):
            return {"total": float("nan"), "adn/total": float("nan")}, {}, ["total", "adn/total"], None
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(train, "_batch_with_precision_recovery", injected)
    with RunLogger("train", log_root=args.log_dir, progress_enabled=False) as run:
        original_info = run.info

        def stop_after_checkpoint(message, *, event="info", **fields):
            original_info(message, event=event, **fields)
            if event == "epoch_completed":
                raise KeyboardInterrupt("test interruption after committed epoch")

        run.info = stop_after_checkpoint
        with pytest.raises(KeyboardInterrupt):
            train._train(args, run)
    checkpoint_path = tmp_path / "run/last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 0 and checkpoint["global_step"] == 1
    first_row = json.loads((tmp_path / "run/metrics.jsonl").read_text().splitlines()[0])
    assert first_row["stability"]["accepted_loss_batches"] == 3
    assert first_row["stability"]["skipped_loss_batches"] == 2
    assert first_row["stability"]["optimizer_updates"] == 1
    assert all(int(s["step"]) == 1 for s in checkpoint["optimizer"]["state"].values())
    # The continuation exercises strict model, optimiser, schedule and RNG load.
    monkeypatch.setattr(train, "_batch_with_precision_recovery", original_batch)
    with RunLogger("train", log_root=args.log_dir, progress_enabled=False) as run:
        result = train._train(args, run)
    assert result["status"] == "ok" and result["resumed"]
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 1 and resumed["global_step"] == 3
    assert resumed["contract"] == checkpoint["contract"]
    assert resumed["scheduler"]["last_epoch"] == 2
    assert args.config.read_bytes() == original_config
    assert (tmp_path / "run/epoch-0001.pt").is_file()
    assert (tmp_path / "run/best.pt").is_file()
    from deepdocforgery.epoch33_preflight import verify

    before = checkpoint_path.read_bytes()
    checked = verify(argparse.Namespace(
        config=args.config, checkpoint=checkpoint_path, device="cpu",
        expected_completed_epochs=2, probe_batches=4, inject_rejected_forward=False,
    ))
    assert checked["config_and_checkpoint_unchanged"]
    assert checked["completed_epochs"] == 2
    assert checked["in_memory_updates_discarded"] == 1
    assert checkpoint_path.read_bytes() == before
    assert args.config.read_bytes() == original_config


def test_preflight_rejects_nonfinite_checkpoint_tensor():
    from deepdocforgery.epoch33_preflight import check_tensors

    with pytest.raises(ValueError, match=r"optimizer\.state\.1\.exp_avg"):
        check_tensors({"state": {1: {"exp_avg": torch.tensor([float("nan")])}}}, "optimizer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_precision_recovery_releases_graph_before_retry():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16-capable GPU required")
    model = ToyModel(large=True).cuda()
    values, _, bad, first = train._batch_with_precision_recovery(
        model, lambda output, _: {"total": output.value.mean()}, toy_batch("cuda"),
        device=torch.device("cuda"), use_amp=True,
        scaler=torch.amp.GradScaler("cuda"), accumulation_steps=4, allow_bf16_retry=True,
    )
    torch.cuda.synchronize()
    assert first is not None and not bad and np.isfinite(values["total"])
    gc.collect()
    assert all(ref() is None for ref in model.saved_refs)

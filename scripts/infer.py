"""Run checkpoint inference and save masks, overlays, instances, and JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.data.manifest import _letterbox
from src.inputLayer.dataModelling import load_yaml, read_jpeg_metadata
from src.inputLayer.freqFeatures import ExactJPEGDCTReader
from src.model import DeepDocForgeryModel
from src.outputLayer.postprocess import extract_instances
from src.training.runtime import load_checkpoint, resolve_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; otherwise use the config embedded in the checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--image-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-instance-area", type=int, default=16)
    parser.add_argument(
        "--exact-jpeg",
        action="store_true",
        help="Use exact coefficients for JPEG inputs at native resolution",
    )
    parser.add_argument(
        "--native-size",
        action="store_true",
        help="Run all images at native resolution instead of configured letterboxing",
    )
    return parser.parse_args()


def _paths(value: Path) -> list[Path]:
    if value.is_file():
        return [value.resolve()]
    if value.is_dir():
        paths = sorted(
            path.resolve()
            for path in value.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if paths:
            return paths
    raise FileNotFoundError(f"No supported images found at {value}")


def _native_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _restore_letterbox(probability: torch.Tensor, transform: object) -> torch.Tensor:
    crop = probability[
        :,
        :,
        transform.top : transform.top + transform.resized_height,
        transform.left : transform.left + transform.resized_width,
    ]
    return F.interpolate(
        crop,
        size=(transform.original_height, transform.original_width),
        mode="bilinear",
        align_corners=False,
    )


def _save_visuals(
    image: Image.Image,
    probability: torch.Tensor,
    output_prefix: Path,
    threshold: float,
) -> tuple[Path, Path, Path]:
    values = probability.squeeze().detach().float().cpu().clamp(0.0, 1.0).numpy()
    probability_path = output_prefix.with_name(output_prefix.name + "-probability.png")
    binary_path = output_prefix.with_name(output_prefix.name + "-mask.png")
    overlay_path = output_prefix.with_name(output_prefix.name + "-overlay.png")
    Image.fromarray(np.round(values * 255.0).astype(np.uint8), mode="L").save(probability_path)
    Image.fromarray((values >= threshold).astype(np.uint8) * 255, mode="L").save(binary_path)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    alpha = (0.65 * values[..., None]).astype(np.float32)
    overlay = rgb * (1.0 - alpha) + red * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(overlay_path)
    return probability_path, binary_path, overlay_path


def main() -> None:
    args = parse_args()
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config is None:
        config = raw_checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Checkpoint has no embedded config; pass --config")
    else:
        config = load_yaml(args.config)
    device = resolve_device(args.device)
    model = DeepDocForgeryModel.from_config(config.get("model", {})).to(device).eval()
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_size = config.get("data", {}).get("image_size", [512, 512])
    exact_reader = ExactJPEGDCTReader() if args.exact_jpeg else None
    records: list[dict[str, object]] = []

    for index, image_path in enumerate(_paths(args.input)):
        with Image.open(image_path) as image_file:
            original = image_file.convert("RGB")
        exact = None
        transform = None
        is_jpeg = image_path.suffix.lower() in {".jpg", ".jpeg"}
        use_native = args.native_size or (args.exact_jpeg and is_jpeg)
        if use_native:
            rgb = _native_tensor(original)
        else:
            rgb, _, _, transform = _letterbox(
                original,
                None,
                (int(configured_size[0]), int(configured_size[1])),
            )
            rgb = rgb.unsqueeze(0)
        if args.exact_jpeg and is_jpeg:
            exact = exact_reader.read(image_path)
            metadata = exact.metadata
        else:
            metadata = read_jpeg_metadata(image_path)
        with torch.no_grad():
            output = model(
                rgb.to(device),
                metadata=metadata.to(device),
                exact_dct=None if exact is None else exact.to(device),
                compute_consistency=exact is not None,
            )
        probability = output.decoder.mask_probability.cpu()
        if transform is not None:
            probability = _restore_letterbox(probability, transform)
        instances = extract_instances(
            probability,
            threshold=args.mask_threshold,
            minimum_area=args.minimum_instance_area,
        )[0]
        output_prefix = output_dir / f"{index:05d}-{image_path.stem}"
        probability_path, binary_path, overlay_path = _save_visuals(
            original, probability, output_prefix, args.mask_threshold
        )
        image_probability = float(output.decoder.image_probability.squeeze().cpu())
        attention = {
            level: {
                branch: float(value.detach().cpu())
                for branch, value in branches.items()
            }
            for level, branches in output.front_end.fusion.mean_attention().items()
        }
        records.append(
            {
                "input": str(image_path),
                "image_probability": image_probability,
                "decision": "forged" if image_probability >= args.image_threshold else "authentic",
                "mask_threshold": args.mask_threshold,
                "exact_jpeg_used": exact is not None,
                "instances": [instance.to_dict() for instance in instances],
                "attention": attention,
                "classification_to_localization_strength": float(
                    output.decoder.classification_to_localization_strength.cpu()
                ),
                "denoising_strength": float(output.decoder.denoising_strength.cpu()),
                "files": {
                    "probability": str(probability_path),
                    "mask": str(binary_path),
                    "overlay": str(overlay_path),
                },
            }
        )
    result = {"status": "ok", "checkpoint": str(args.checkpoint.resolve()), "results": records}
    report = output_dir / "predictions.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "images": len(records), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()

# Epoch 33: evidence and repair

## 27 October follow-up: validation AUROC loop

Epoch 33 completed all 9,853 training batches and 1,737 validation batches. The repaired training path recovered 28 non-finite FP16 losses through BF16 and avoided OOM. Two isolated non-finite gradient steps were skipped without consecutive escalation. After validation, `py-spy --locals` proved the main thread was looping in `binary_auroc`: `threshold` was `NaN` and `index` remained 29. The old inner loop advanced only while `score == threshold`; IEEE NaN is unequal to itself, so it could never advance.

Version 2 prevents that loop by returning an unavailable AUROC when any score is non-finite and recording `image/non_finite_scores`. More importantly, validation now checks both prediction heads and every loss component before metrics consume them. A non-finite FP16 validation batch is released and retried in BF16. If BF16 also fails, evaluation stops before checkpoint selection uses invalid predictions.

The one-time GDB rescue changes only the already-running AUROC score list, allowing the trained epoch-33 weights to reach the atomic checkpoint boundary. Its image AUROC is invalid and must not be used. After that checkpoint is committed, stop before doing substantive epoch-34 work, install v2, and resume from the verified epoch-33 `last.pt`.

This repair is based on the supplied archive, every training log pair, model metrics, and GitHub history through 536c54eb347629414c520f01cf3f43553c040fc7. It preserves the original YAML, checkpoint schema and model parameter layout. It has not completed epoch 33 on the university GPU: neither that GPU, the real checkpoint, nor the underlying training images was available here.

## What is established

The main metrics contain epochs 1–32. Run `20261019T031447Z-8041-36ecd1` resumed epoch 32, saved its last checkpoint, then failed in epoch 33. Later attempts explicitly resumed after 32/50 at global step 78,848 with RNG restored. The epoch-32 loss was 0.8028527459287165; validation pixel_macro/f1_forged was 0.8041331341444312. The historical best remained 0.8114558513227331 (epoch 11). Continue from `last.pt`, not `best.pt`. The supplied model `status.json` was an old epoch-17 snapshot and is not the latest completed-epoch evidence.

The original 19 October failure is `Non-finite loss at epoch 32, batch 5887`. Those fields were zero-based: this is human epoch 33, batch 5,888. The subsequent code tried to skip bad forward losses. All four logged uses of that path immediately hit OOM:

| Run | Human bad batch | Warning → OOM timestamps labelled UTC | Allocated / unused reserved at OOM | Allocation that failed |
|---|---:|---|---|---:|
| 20261019T092634Z-163180-c4dff3 | 5,906 | 10:54:48 → 10:54:49 | 18.48 GiB / 22.78 MiB | 20 MiB |
| 20261020T051522Z-1675-9c8341 | 8,815 | 07:26:09 → 07:26:10 | 18.97 GiB / 47.97 MiB | 12 MiB |
| 20261021T050446Z-1713-afce2d | 6,214 | 06:37:24 → 06:37:24 | 18.83 GiB / 126.41 MiB | 88 MiB |
| 20261026T033246Z-7850-c03172 | 6,069 | 05:03:54 → 05:03:54 | 18.50 GiB / 55.59 MiB | 22 MiB |

Exact JSONL evidence: 19 October lines 18–19; 20 October 20–21; 21 October 21–22; 26 October 19–20, in the run-named JSONL files. The last run had only 21.06 MiB free at the failed 22 MiB request.

The faulty control flow retains the rejected forward's `output` and `losses`. Because backward never ran, saved activations remain live. At `output = model(...)` on the next iteration, Python evaluates the new forward before replacing the old output. Two sets of activations overlap. The loss dictionary also keeps the previous graph reachable. `empty_cache()` cannot release live tensors. The very small unused-reserved amounts in these errors do not support allocator fragmentation as the primary explanation.

The repair confines forward, loss inspection, backward and attention extraction to a function returning only Python scalars. Every exit releases its graph before the next forward. A regression test checks that graph tensors become unreachable on both successful and rejected batches. The GPU probe can deliberately reject a real full-size forward and retry it, testing this path on the target hardware.

## What remains unproven

The old logs record total non-finite loss and sample IDs, not component losses or intermediate activations. They cannot identify which branch first overflowed, or whether a particular image caused it. Bad-batch positions and sample lists differ across attempts. It would be inaccurate to declare a specific sample corrupt from this evidence.

Direct half-precision tests expose fragile reductions in ADN zero construction/prototypes and decoder weighted pooling. The repair performs these small reductions in FP32 and forms a differentiable zero without summing logits first. However, CUDA autocast already promotes many reductions, including sum and normalisation, to FP32. A direct CPU-half overflow is therefore NOT proof of the original CUDA NaN's cause. These are defensive numerical fixes, not a claimed diagnosis of that first non-finite value.

On a CUDA device reporting BF16 support, an FP16 forward with any non-finite loss component is retried once on the same batch in BF16. Model RNG state is replayed; already accumulated gradients and the AMP scale are retained. The failed graph is released before retry. BF16 uses 16-bit storage and a wider exponent range, but this does not guarantee identical kernel workspace requirements: run the GPU probe. Normal batches still use the original FP16 AMP configuration. This is a deliberate numerical-recovery policy change even though the YAML is unchanged; the continuation is not claimed to be bit-for-bit identical to unrepaired training.

If the retry also fails, the existing bounded skip policy remains: log component losses and sample identities, keep prior valid gradients, and stop after five consecutive non-finite-loss batches by default. Non-finite gradient steps are blocked before optimiser weights or Adam moments can change; twenty consecutive such steps stop the run by default. Neither policy can cure sustained model divergence. No promise of an epoch-33 completion is made without running it.

## Memory and accumulation decisions

- Retain physical batch 11, image size 512×512, accumulation 4, nominal effective batch 44, all branches/heads and gradient checkpointing.
- Load the checkpoint onto CPU, copy model state and restore optimiser state onto parameter devices, then release the saved payload. This avoids retaining an extra saved model copy in VRAM. Do not map saved RNG tensors to GPU.
- Preserve scale throughout an accumulation window. Never lower the GradScaler scale in response to a failed forward while older scaled gradients are pending.
- Count accepted microbatches for accumulation; flush pending valid gradients at epoch end even if the last fetched batch was rejected. Preserve the existing division by four for a partial final window. This avoids silently losing the tail.
- Preserve historical `global_step` meaning (step attempts, including skipped AMP steps). Record successful updates separately. Skipped batches can change the final step count; do not assert that epoch 33 must end at an exact predetermined global step.
- Record per-epoch accepted/skipped loss batches, skipped gradient steps, successful optimiser updates, BF16 retries and recoveries. New epoch/batch log fields are human one-based; `batch_index` records the zero-based index explicitly.

The physical 20,475 MiB is about 19.995 GiB. The logs report PyTorch-visible total 19.54 GiB and regular progress peaks around 16.897 GiB allocated / 17.439 GiB reserved. Desktop use and CUDA/non-PyTorch memory still consume capacity. This supports removing the extra live graph before reducing the already-working batch. Cached `run_failed.resources` is the last progress snapshot, not a fresh failure measurement; the OOM exception's numbers are the relevant measurements.

## Earlier failures traced separately

Across 51 runs there are 2,846 JSON events, 21 recorded run-failure events, 93 non-finite-gradient warnings, and four non-finite-loss skip warnings. Nineteen runs lack a terminal event; their stopping cause is unknown. There are 91 epoch-completed events across main training and HPO, not 91 main epochs.

| Failure class | Count | Interpretation and disposition |
|---|---:|---|
| Startup OOM, 16 September | 5 | Batch 12, accumulation 4, effective 48 in the trial plan. Distinct from the late batch-11 epoch-33 failure. |
| OOM immediately after rejected loss, 19–26 October | 4 | Graph-lifetime defect addressed here. |
| Missing processed dataset paths | 3 | 16/17 September and 26 October; consistent with inaccessible/unmounted processed storage. Restore the same mount and paths; do not regenerate the manifest. |
| Changed total-epoch count | 2 | 5 October; cosine scheduler correctly rejected a different total. Keep 50 epochs and omit `--epochs`. |
| RNG state must be ByteTensor | 1 | 5 October; addressed in commit 770e6f5 with CPU uint8 conversion. Retained here. |
| Broken PNG chunk | 1 | 5 October; Pillow raised SyntaxError while decoding, not a Python source syntax error. The upload does not establish which file was repaired or why subsequent runs progressed. No broad data rewrite is justified. |
| Original non-finite loss | 1 | 19 October, human epoch 33/batch 5,888. Numerical origin unproven; new component diagnostics and bounded precision recovery address recurrence. |
| KeyboardInterrupt | 4 | Explicit interruption, not an inferred software defect. |

All 51 runs, including incomplete starts and every HPO trial, appear individually in ALL_TRAINING_RUNS.md. The supplied text logs and JSON event streams were both included. FILE_INVENTORY.md accounts for all 231 original files; COMMIT_HISTORY.md includes all 28 commits.

## Time interpretation

JSON timestamps explicitly claim UTC; text logs use local time. Convert UTC to Pakistan by adding five hours once: for example, the final run's 03:32:46Z start corresponds to 08:32:46 PKT, consistent with a morning university start. Do not add another five hours to an already-local value. The user's occasional physical clock error is separate from timezone conversion and cannot be reconstructed precisely from these logs alone. Within-run ordering and elapsed warning-to-OOM intervals are reliable evidence even if wall-clock time is wrong. Git diffs are compared by observed behaviour, not assigned mechanically by calendar timestamp.

## Source and validation boundaries

Source paths examined include training, checkpoint/RNG/scheduler, telemetry, data transforms and sampling, model branches, fusion, decoder, objectives/losses, metrics, preparation, doctor, HPO, evaluation, inference and status. Source tests cover orientation, grouping, official subset boundaries, target masks, exact/fallback DCT, model connectivity and resume semantics. No configuration, dataset boundary, manifest, or checkpoint weight is shipped in this repair.

`configs/cuda/full.yaml` and its README contain historical settings distinct from the active `HPO_RESULTS.yaml`. Resume uses the latter: learning rate 0.0001, epochs 50, batch 11, accumulation 4. Do not substitute full.yaml, an HPO trial YAML, or a regenerated manifest. The continuation contract includes the resolved manifest path, not just its bytes: keep the existing repository/data location.

The preflight validates the actual saved model and optimiser tensors, scheduler epoch, full resume fields and current contract. A failed contract check is not permission to erase the contract or reset the optimiser. Stop and compare the checkpoint's embedded configuration, current original YAML and manifest identity.

Official PyTorch references checked for this repair:

- AMP accumulation: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-accumulation
- CUDA autocast operator policies: https://docs.pytorch.org/docs/2.11/amp.html#cuda-op-specific-behavior
- CUDA allocator/live tensor distinction: https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management

The tested CPU build is torch 2.11.0+cpu. The supplied GPU logs identify torch 2.11.0+cu128, CUDA 12.8, Python 3.12.3. Keep the existing university virtual environment; do not upgrade dependencies as part of this repair.

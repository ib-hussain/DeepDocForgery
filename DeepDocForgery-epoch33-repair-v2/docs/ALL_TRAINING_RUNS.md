# Every supplied training run

Each row accounts for BOTH the `.log` and `.jsonl` file with this run ID in `output/logs/train/`. All 51 pairs were parsed; this table is an index of derived findings, not a replacement raw log.

PKT below is the conversion of the JSON timestamp labelled UTC, plus five hours. It is not a claim that the physical clock was accurate. Epochs/batches shown here are human, one-based numbers. `No terminal event` does not establish why execution stopped.

| Run ID | PKT start from declared UTC | Completed epochs in this run | Last training progress | Gradient warnings | Outcome |
|---|---|---|---|---:|---|
| `20260915T095243Z-22157-667139` | 2026-09-15T14:52:43+05:00 | — | — | 0 | No terminal event; cause not recorded |
| `20260915T095513Z-25074-0708ca` | 2026-09-15T14:55:13+05:00 | — | E1 10839/10839 | 5 | No terminal event; cause not recorded |
| `20260916T030925Z-7018-ff84fc` | 2026-09-16T08:09:25+05:00 | — | — | 0 | KeyboardInterrupt |
| `20260916T031018Z-8126-15ae50` | 2026-09-16T08:10:18+05:00 | — | — | 0 | Processed data paths unavailable; check mount |
| `20260916T031034Z-8338-53bb3f` | 2026-09-16T08:10:34+05:00 | — | — | 0 | KeyboardInterrupt |
| `20260916T031123Z-8693-d47b4d` | 2026-09-16T08:11:23+05:00 | — | — | 0 | No terminal event; cause not recorded |
| `20260916T031342Z-10598-f13b2d` | 2026-09-16T08:13:42+05:00 | — | — | 0 | Startup OOM (batch 12) |
| `20260916T031600Z-12618-3ec72a` | 2026-09-16T08:16:01+05:00 | — | — | 0 | Startup OOM (batch 12) |
| `20260916T031749Z-14461-d30bf5` | 2026-09-16T08:17:49+05:00 | — | — | 0 | Startup OOM (batch 12) |
| `20260916T031946Z-16231-7dc571` | 2026-09-16T08:19:46+05:00 | — | — | 0 | Startup OOM (batch 12) |
| `20260916T032127Z-17816-7a0c62` | 2026-09-16T08:21:27+05:00 | — | — | 0 | Startup OOM (batch 12) |
| `20260916T032308Z-19696-382a44` | 2026-09-16T08:23:08+05:00 | — | — | 0 | No terminal event; cause not recorded |
| `20260916T032410Z-21167-6f9274` | 2026-09-16T08:24:10+05:00 | — | — | 0 | No terminal event; cause not recorded |
| `20260916T032546Z-22795-5526b6` | 2026-09-16T08:25:46+05:00 | — | — | 0 | KeyboardInterrupt |
| `20260916T032607Z-23331-c31221` | 2026-09-16T08:26:07+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 3 | HPO trial completed |
| `20260916T040849Z-56172-579088` | 2026-09-16T09:08:49+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T045252Z-93992-cb8c71` | 2026-09-16T09:52:52+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T053553Z-131344-8854ca` | 2026-09-16T10:35:53+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 3 | HPO trial completed |
| `20260916T061930Z-168894-1ce376` | 2026-09-16T11:19:30+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T070245Z-206344-b8aa9d` | 2026-09-16T12:02:46+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T074602Z-243746-064aec` | 2026-09-16T12:46:02+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 4 | HPO trial completed |
| `20260916T082921Z-281285-7e6ed9` | 2026-09-16T13:29:21+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T091245Z-318662-bff9aa` | 2026-09-16T14:12:45+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 3 | HPO trial completed |
| `20260916T095543Z-356563-ca0211` | 2026-09-16T14:55:43+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 2 | HPO trial completed |
| `20260916T103842Z-393964-7a668e` | 2026-09-16T15:38:42+05:00 | 1, 2, 3, 4, 5 | E5 400/400 | 3 | HPO trial completed |
| `20260916T112149Z-431166-2fdb59` | 2026-09-16T16:21:49+05:00 | 1, 2, 3, 4 | E5 300/400 | 2 | No terminal event; cause not recorded |
| `20260917T050224Z-10619-c3b649` | 2026-09-17T10:02:24+05:00 | — | — | 0 | Processed data paths unavailable; check mount |
| `20260917T050720Z-14771-346ca6` | 2026-09-17T10:07:20+05:00 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | E10 9853/9853 | 16 | No terminal event; cause not recorded |
| `20261005T032454Z-9557-1d0d60` | 2026-10-05T08:24:54+05:00 | — | — | 0 | Scheduler rejected changed total epochs |
| `20261005T033115Z-12940-395e72` | 2026-10-05T08:31:16+05:00 | — | — | 0 | Scheduler rejected changed total epochs |
| `20261005T034529Z-17657-5522bb` | 2026-10-05T08:45:29+05:00 | — | — | 0 | RNG restore CPU/uint8 bug |
| `20261005T035649Z-23712-5e0589` | 2026-10-05T08:56:49+05:00 | — | — | 0 | Pillow broken PNG chunk in DataLoader |
| `20261005T045426Z-63068-a71579` | 2026-10-05T09:54:26+05:00 | 10, 11 | E12 4000/9853 | 2 | No terminal event; cause not recorded |
| `20261006T062414Z-12300-bf42f2` | 2026-10-06T11:24:14+05:00 | — | — | 0 | KeyboardInterrupt |
| `20261006T062841Z-16324-24b3df` | 2026-10-06T11:28:41+05:00 | 12 | E13 9853/9853 | 2 | No terminal event; cause not recorded |
| `20261007T033305Z-1736-5fbe45` | 2026-10-07T08:33:05+05:00 | 13, 14 | E15 9853/9853 | 5 | No terminal event; cause not recorded |
| `20261008T050914Z-7837-729684` | 2026-10-08T10:09:14+05:00 | 15, 16 | E17 3000/9853 | 3 | No terminal event; cause not recorded |
| `20261012T050205Z-9913-c6732b` | 2026-10-12T10:02:05+05:00 | 17, 18 | E19 4000/9853 | 2 | No terminal event; cause not recorded |
| `20261013T050534Z-1762-4ba9a5` | 2026-10-13T10:05:34+05:00 | 19, 20 | E21 4000/9853 | 2 | No terminal event; cause not recorded |
| `20261014T050113Z-9888-853c22` | 2026-10-14T10:01:13+05:00 | 21, 22 | E23 3000/9853 | 2 | No terminal event; cause not recorded |
| `20261015T045902Z-9087-6ad411` | 2026-10-15T09:59:02+05:00 | 23 | E24 9853/9853 | 2 | No terminal event; cause not recorded |
| `20261016T045710Z-9028-bece4d` | 2026-10-16T09:57:10+05:00 | 24 | E25 5000/9853 | 2 | No terminal event; cause not recorded |
| `20261016T105531Z-6792-e1a000` | 2026-10-16T15:55:31+05:00 | 25, 26, 27, 28, 29, 30 | E31 2000/9853 | 6 | No terminal event; cause not recorded |
| `20261017T132214Z-9176-fe094e` | 2026-10-17T18:22:14+05:00 | 31 | E32 2000/9853 | 1 | No terminal event; cause not recorded |
| `20261019T031447Z-8041-36ecd1` | 2026-10-19T08:14:47+05:00 | 32 | E33 5000/9853 | 4 | Original non-finite forward; old code stopped |
| `20261019T092634Z-163180-c4dff3` | 2026-10-19T14:26:34+05:00 | — | E33 5000/9853 | 2 | OOM immediately after rejected loss |
| `20261020T051522Z-1675-9c8341` | 2026-10-20T10:15:22+05:00 | — | E33 8000/9853 | 1 | OOM immediately after rejected loss |
| `20261021T033646Z-8491-328faa` | 2026-10-21T08:36:46+05:00 | — | E33 1000/9853 | 0 | No terminal event; cause not recorded |
| `20261021T050446Z-1713-afce2d` | 2026-10-21T10:04:46+05:00 | — | E33 6000/9853 | 4 | OOM immediately after rejected loss |
| `20261026T033216Z-7618-d40d34` | 2026-10-26T08:32:16+05:00 | — | — | 0 | Processed data paths unavailable; check mount |
| `20261026T033246Z-7850-c03172` | 2026-10-26T08:32:46+05:00 | — | E33 6000/9853 | 2 | OOM immediately after rejected loss |

## Non-training logs

- `doctor/20260909T055304Z-18614-825740`: ten events; four checks completed; no warning/error event.
- `prepare/20260909T061525Z-31968-474514`: 194 events; preparation completed. The warning about 8,835 masked-perceptual proxy collision groups is a dataset-protocol diagnostic. Official test membership took precedence; this is not an epoch-33 CUDA failure.
- `hpo/20260916T032605Z-23240-3a340f`: 27 events, twelve trials started, eleven completed. Trial twelve stops during its fifth epoch with no terminal event. Those trial epochs are separate from the main 32-epoch run.
- `status/20260909T033844Z-8357-09ed3a` and `status/20260909T055605Z-19186-2477bc`: four events each; attention-required snapshots of prior pipeline state, not training exceptions.
- Setup/environment text logs, latest pointers, HPO result files, model metrics/status and preparation metadata are catalogued in FILE_INVENTORY.md. They were checked as supporting evidence; none is substituted for the latest model checkpoint.

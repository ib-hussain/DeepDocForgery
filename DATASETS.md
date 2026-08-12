# Dataset acquisition and preparation

DeepDocForgery code is GPL-3.0-or-later. That does not grant permission to
redistribute third-party datasets. Keep downloaded data outside commits and
record the exact dataset release, terms, checksum, and access date used in each
experiment.

## 1. Included synthetic demo

```bash
python -m scripts.make_demo_dataset --output data/demo --groups 12 --force
```

This creates 24 small images, masks, degradation labels, and a manifest. It is
licence-clean and sufficient for tests, but intentionally unsuitable for
scientific claims.

## 2. DocTamper

The [official DocTamper repository](https://github.com/qcf-568/DocTamper)
states that the dataset is non-commercial, requires an academic application,
and may require a university or research-institute affiliation. Apply and
accept its terms before downloading.

After authorization, the official Kaggle listing is
`dinmkeljiame/doctamper`. Install the official CLI and authenticate it in
your own terminal:

```bash
python -m pip install -e ".[data,download]"
kaggle datasets download \
  -d dinmkeljiame/doctamper \
  -p data/raw/doctamper \
  --unzip
```

Do not paste a Kaggle token into source code or chat. Once an extracted subset
directory contains `data.mdb`, export it:

```bash
python -m scripts.prepare_doctamper \
  --input data/raw/doctamper/DocTamperV1-FCD \
  --output data/processed/doctamper-fcd
```

The exporter:

- supports zero- or one-based LMDB keys,
- preserves JPEG bytes for exact-DCT use,
- normalizes masks to binary PNG,
- creates train/validation/test records, and
- never imports DocTamper model source.

Important limitation: the public LMDB key contract does not expose a clean
source-document family id. The exporter therefore cannot prove that related
derivatives are separated. Audit upstream provenance and replace
`source_group` before publishing split-sensitive results.

## 3. MIDV-DM

The [MIDV-DM paper page](https://computeroptics.ru/eng/KO/Annot/KO49-6/490625e.html)
describes 1,000 original and 8,000 manipulated identity-document images with
pixel masks and annotations. Use the official Smart Engines dataset page or
the release location supplied by the authors; no stable unattended archive URL
is hard-coded because the official distribution location may change.

After downloading and extracting images/masks:

```bash
python -m scripts.build_manifest \
  --images data/raw/midv-dm/images \
  --masks data/raw/midv-dm/masks \
  --output data/processed/midv-dm/manifest.jsonl \
  --dataset midv-dm \
  --group-regex '^(.*?)(?:[_-](?:fake|forged|tampered).*)?$'
```

Adjust the regular expression to the actual release naming scheme. Its first
capture group must identify all derivatives of one original document.

## 4. DanceText

The [official DanceText repository](https://github.com/qcf-568/DanceText)
should be treated as the only authoritative release channel. As of the
2026-08-12 project review, no stable automated dataset download and no DS-Net
model source were available there. Do not scrape mirrors or invent a URL.

When authorized data is released as normal image/mask folders, use
`scripts.build_manifest`. If it uses another container, add a small exporter
that emits the same JSONL contract without changing model code.

## 5. Any paired image/mask dataset

```bash
python -m scripts.build_manifest \
  --images /path/to/images \
  --masks /path/to/masks \
  --output data/processed/my-dataset/manifest.jsonl \
  --dataset my-dataset \
  --group-regex '^(source-[0-9]+)'
```

Masks are matched by stem. Use `--mask-suffix _mask` when
`document-1.jpg` corresponds to `document-1_mask.png`.

If a folder also contains authentic images without masks:

```bash
python -m scripts.build_manifest \
  --images /path/to/images \
  --masks /path/to/masks \
  --output data/processed/my-dataset/manifest.jsonl \
  --dataset my-dataset \
  --allow-unmasked-authentic
```

Those records supervise classification only.

## 6. Merge prepared datasets

```bash
python -m scripts.merge_manifests \
  --input data/processed/doctamper-fcd/manifest.jsonl \
  --input data/processed/midv-dm/manifest.jsonl \
  --output data/processed/combined/manifest.jsonl
```

The command re-roots paths and namespaces sample/group identifiers. It
preserves each source manifest's split; it does not hide a poor upstream split.

## 7. Split requirements

- Assign splits by `source_group`, never by final image filename alone.
- Keep an original document and every manipulated derivative in one split.
- Do not tune thresholds on the test split.
- Report per-dataset results as well as combined results.
- Record real/authentic balance and manipulation-family balance.
- For cross-generator evaluation, hold out generator families, not random
  images from the same generator.

The loader rejects a manifest when one `source_group` appears in multiple
splits.

## 8. Storage on WSL

Code under `/mnt/d` works, but large random-read training is usually faster
inside the WSL Linux filesystem. Keep reproducible manifests in the repository
and point their paths at a Linux-side dataset location when training.

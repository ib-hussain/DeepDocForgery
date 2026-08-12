"""Generate a tiny licence-clean document-forgery dataset on local disk."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files with matching names",
    )
    return parser.parse_args()


def _split(group_index: int, groups: int) -> str:
    train_end = max(1, round(groups * 2 / 3))
    val_end = max(train_end + 1, round(groups * 5 / 6))
    if group_index < train_end:
        return "train"
    if group_index < val_end:
        return "val"
    return "test"


def _base_document(group_index: int, rng: random.Random) -> Image.Image:
    width, height = 192, 128
    image = Image.new("RGB", (width, height), (rng.randint(238, 250),) * 3)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, width - 6, height - 6), outline=(80, 90, 105), width=1)
    draw.rectangle((12, 12, 44, 43), fill=(185, 196, 207), outline=(65, 75, 90))
    draw.text((52, 14), "NORTHSTAR DOCUMENT", fill=(20, 28, 38))
    draw.text((52, 29), f"RECORD {group_index:04d}", fill=(45, 52, 62))
    fields = (
        ("NAME", f"RESEARCH SUBJECT {group_index:02d}"),
        ("DATE", f"{1 + group_index % 27:02d}-08-2026"),
        ("VALUE", f"PKR {1200 + group_index * 137}.00"),
        ("CODE", f"DD-{group_index:04d}-{rng.randint(100, 999)}"),
    )
    for row, (name, value) in enumerate(fields):
        y = 53 + row * 15
        draw.text((14, y), f"{name}:", fill=(55, 60, 68))
        draw.text((61, y), value, fill=(25, 30, 38))
        draw.line((12, y + 11, width - 13, y + 11), fill=(205, 208, 212), width=1)
    return image


def _save_jpeg(
    image: Image.Image,
    destination: Path,
    *,
    quality: int,
    double_compression: bool,
) -> None:
    if double_compression:
        intermediate = io.BytesIO()
        image.save(intermediate, format="JPEG", quality=min(98, quality + 13), subsampling=2)
        intermediate.seek(0)
        with Image.open(intermediate) as reopened:
            reopened.convert("RGB").save(
                destination, format="JPEG", quality=quality, subsampling=2
            )
    else:
        image.save(destination, format="JPEG", quality=quality, subsampling=2)


def main() -> None:
    args = parse_args()
    if args.groups < 6:
        raise ValueError("Use at least six source groups so every split is populated")
    output = args.output.resolve()
    images = output / "images"
    masks = output / "masks"
    manifest = output / "manifest.jsonl"
    if manifest.exists() and not args.force:
        raise FileExistsError(f"{manifest} already exists; pass --force to regenerate")
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []

    for group_index in range(args.groups):
        source_group = f"document-{group_index:04d}"
        split = _split(group_index, args.groups)
        base = _base_document(group_index, rng)
        for forged in (False, True):
            sample_id = f"{source_group}-{'forged' if forged else 'authentic'}"
            image = base.copy()
            mask = Image.new("L", image.size, 0)
            quality = 72 + (group_index * 7 + int(forged) * 5) % 24
            double_compression = bool(forged and group_index % 2 == 0)
            tamper_type = "none"
            if forged:
                draw = ImageDraw.Draw(image)
                mask_draw = ImageDraw.Draw(mask)
                row = group_index % 4
                top = 50 + row * 15
                rectangle = (58, top, 178, top + 12)
                background = image.getpixel((180, top + 2))
                draw.rectangle(rectangle, fill=background)
                replacement = (
                    f"ALTERED {9000 + group_index * 31}"
                    if row != 2
                    else f"PKR {8900 + group_index * 211}.00"
                )
                draw.text((61, top + 1), replacement, fill=(31, 35, 43))
                mask_draw.rectangle(rectangle, fill=255)
                tamper_type = "text_replacement"

            image_path = images / f"{sample_id}.jpg"
            mask_path = masks / f"{sample_id}.png"
            _save_jpeg(
                image,
                image_path,
                quality=quality,
                double_compression=double_compression,
            )
            mask.save(mask_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "image": image_path.relative_to(output).as_posix(),
                    "mask": mask_path.relative_to(output).as_posix(),
                    "split": split,
                    "label": int(forged),
                    "source_group": source_group,
                    "dataset": "deepdocforgery-demo",
                    "tamper_type": tamper_type,
                    "jpeg_quality": quality,
                    "double_compression": double_compression,
                    "noise_type": "none",
                    "noise_strength": 0.0,
                }
            )

    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    readme = output / "README.md"
    readme.write_text(
        "# DeepDocForgery demo data\n\n"
        "These synthetic documents and masks are generated entirely by "
        "`scripts.make_demo_dataset`; they contain no external images or personal data. "
        "They exist only for pipeline tests and must not be used as scientific evidence.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "samples": len(records),
                "groups": args.groups,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

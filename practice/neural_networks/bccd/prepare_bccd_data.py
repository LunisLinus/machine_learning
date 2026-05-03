from __future__ import annotations

import argparse
from pathlib import Path

from bccd_utils import apply_splits, build_annotation_tables, load_split_ids, prepare_yolo_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BCCD tables and YOLO labels.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "BCCD_Dataset",
        help="Path to the downloaded BCCD_Dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory where prepared tables and YOLO labels will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    boxes_df, counts_df = build_annotation_tables(args.dataset_dir)
    split_ids = load_split_ids(args.dataset_dir)
    counts_with_split = apply_splits(counts_df, split_ids)

    metadata_dir = args.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    boxes_path = metadata_dir / "boxes.csv"
    counts_path = metadata_dir / "image_counts.csv"
    split_counts_path = metadata_dir / "split_summary.csv"

    boxes_df.to_csv(boxes_path, index=False)
    counts_with_split.to_csv(counts_path, index=False)
    counts_with_split.groupby("split")[["RBC", "WBC", "Platelets", "total_cells"]].agg(["count", "sum"]).to_csv(
        split_counts_path
    )

    yolo_dir = args.output_dir / "yolo_dataset"
    yaml_path = prepare_yolo_dataset(args.dataset_dir, yolo_dir, split_ids)

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "num_images": int(len(counts_with_split)),
        "num_boxes": int(len(boxes_df)),
        "splits": {split_name: len(filenames) for split_name, filenames in split_ids.items()},
        "class_totals": {
            class_name: int(counts_with_split[class_name].sum()) for class_name in ("RBC", "WBC", "Platelets")
        },
        "boxes_csv": str(boxes_path.resolve()),
        "counts_csv": str(counts_path.resolve()),
        "yolo_yaml": str(yaml_path.resolve()),
    }
    write_json(summary, metadata_dir / "prepare_summary.json")

    print("Prepared BCCD dataset artifacts:")
    print(f"  boxes: {boxes_path}")
    print(f"  counts: {counts_path}")
    print(f"  yolo yaml: {yaml_path}")


if __name__ == "__main__":
    main()

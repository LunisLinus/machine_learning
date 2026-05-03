from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bccd_utils import CLASSES, compute_count_metrics, load_split_ids, prepare_yolo_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on BCCD and count cells from detections.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "BCCD_Dataset",
        help="Path to the downloaded BCCD_Dataset directory.",
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "metadata" / "image_counts.csv",
        help="Prepared CSV with image-level counts produced by prepare_bccd_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "detection",
        help="Directory where YOLO outputs and count metrics will be stored.",
    )
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="bccd_yolo")
    parser.add_argument(
        "--augmentation",
        choices=("none", "mild", "strong"),
        default="strong",
        help="YOLO train-time augmentation preset. Strong is useful for the small BCCD dataset.",
    )
    return parser.parse_args()


def yolo_augmentation_kwargs(preset: str) -> dict[str, float | int]:
    if preset == "none":
        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 0,
        }

    if preset == "mild":
        return {
            "hsv_h": 0.01,
            "hsv_s": 0.35,
            "hsv_v": 0.25,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.25,
            "shear": 1.0,
            "perspective": 0.0,
            "flipud": 0.1,
            "fliplr": 0.5,
            "mosaic": 0.4,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 10,
        }

    return {
        "hsv_h": 0.015,
        "hsv_s": 0.55,
        "hsv_v": 0.35,
        "degrees": 12.0,
        "translate": 0.08,
        "scale": 0.35,
        "shear": 2.0,
        "perspective": 0.0005,
        "flipud": 0.25,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.08,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    }


def extract_detection_metrics(validation_result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if hasattr(validation_result, "box"):
        metrics["map50"] = float(getattr(validation_result.box, "map50", np.nan))
        metrics["map50_95"] = float(getattr(validation_result.box, "map", np.nan))
        metrics["map75"] = float(getattr(validation_result.box, "map75", np.nan))
    return metrics


def predict_counts(model: Any, image_paths: list[str], conf_threshold: float, iou_threshold: float, device: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for image_path in image_paths:
        result = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False,
        )[0]

        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.array([])
        counts = {class_name: 0 for class_name in CLASSES}
        for class_id in class_ids:
            counts[CLASSES[class_id]] += 1

        records.append({"filename": Path(image_path).name, **counts})

    return pd.DataFrame(records).sort_values("filename").reset_index(drop=True)


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "The ultralytics package is required. Install dependencies from requirements.txt before running."
        ) from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_ids = load_split_ids(args.dataset_dir)
    yolo_dir = args.output_dir / "yolo_dataset"
    yaml_path = prepare_yolo_dataset(args.dataset_dir, yolo_dir, split_ids)

    model = YOLO(args.model)
    augmentation_kwargs = yolo_augmentation_kwargs(args.augmentation)
    train_result = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        seed=args.seed,
        project=str(args.output_dir / "runs"),
        name=args.run_name,
        exist_ok=True,
        **augmentation_kwargs,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("YOLO trainer is not available after training.")

    best_weights = Path(trainer.best if trainer.best.exists() else trainer.last)
    trained_model = YOLO(str(best_weights))

    validation_result = trained_model.val(
        data=str(yaml_path),
        split="test",
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        device=args.device,
    )

    counts_df = pd.read_csv(args.counts_csv)
    test_df = counts_df.loc[counts_df["split"] == "test", ["filename", "image_path", *CLASSES]].copy()
    pred_df = predict_counts(
        trained_model,
        image_paths=test_df["image_path"].tolist(),
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )

    merged_df = test_df.merge(pred_df, on="filename", suffixes=("_true", "_pred")).sort_values("filename")
    true_matrix = merged_df[[f"{class_name}_true" for class_name in CLASSES]].to_numpy(dtype=np.int32)
    pred_matrix = merged_df[[f"{class_name}_pred" for class_name in CLASSES]].to_numpy(dtype=np.int32)

    count_metrics = compute_count_metrics(true_matrix, pred_matrix, class_names=CLASSES)
    merged_df.to_csv(args.output_dir / "test_count_predictions.csv", index=False)
    write_json(
        {
            "weights": str(best_weights.resolve()),
            "augmentation": {
                "preset": args.augmentation,
                "kwargs": augmentation_kwargs,
            },
            "train_metrics": extract_detection_metrics(train_result),
            "dataset_yaml": str(yaml_path.resolve()),
            "detection_metrics": extract_detection_metrics(validation_result),
            "count_metrics": count_metrics,
        },
        args.output_dir / "metrics.json",
    )

    print(f"Detection weights: {best_weights}")
    print(f"Count predictions: {args.output_dir / 'test_count_predictions.csv'}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CLASSES = ("RBC", "WBC", "Platelets")
CLASS_TO_ID = {class_name: idx for idx, class_name in enumerate(CLASSES)}


def parse_voc_annotation(xml_path: Path) -> dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if filename is None:
        raise ValueError(f"Missing filename in annotation: {xml_path}")

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size block in annotation: {xml_path}")

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        bbox = obj.find("bndbox")
        if class_name is None or bbox is None:
            continue

        xmin = int(bbox.findtext("xmin", default="0"))
        ymin = int(bbox.findtext("ymin", default="0"))
        xmax = int(bbox.findtext("xmax", default="0"))
        ymax = int(bbox.findtext("ymax", default="0"))

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width": xmax - xmin,
                "bbox_height": ymax - ymin,
                "bbox_area": (xmax - xmin) * (ymax - ymin),
            }
        )

    return {
        "filename": filename,
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }


def build_annotation_tables(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations_dir = dataset_dir / "BCCD" / "Annotations"
    image_dir = dataset_dir / "BCCD" / "JPEGImages"

    box_records: list[dict[str, Any]] = []
    count_records: list[dict[str, Any]] = []

    xml_files = sorted(annotations_dir.glob("*.xml"))
    for xml_path in xml_files:
        parsed = parse_voc_annotation(xml_path)
        filename = parsed["filename"]
        image_path = image_dir / filename

        counts = {class_name: 0 for class_name in CLASSES}
        for obj in parsed["objects"]:
            counts[obj["class_name"]] += 1
            box_records.append(
                {
                    "filename": filename,
                    "image_path": str(image_path.resolve()),
                    "image_width": parsed["image_width"],
                    "image_height": parsed["image_height"],
                    **obj,
                }
            )

        count_records.append(
            {
                "filename": filename,
                "image_path": str(image_path.resolve()),
                "image_width": parsed["image_width"],
                "image_height": parsed["image_height"],
                **counts,
                "total_cells": sum(counts.values()),
            }
        )

    boxes_df = pd.DataFrame(box_records).sort_values(["filename", "class_name"]).reset_index(drop=True)
    counts_df = pd.DataFrame(count_records).sort_values("filename").reset_index(drop=True)
    return boxes_df, counts_df


def load_split_ids(dataset_dir: Path) -> dict[str, list[str]]:
    split_dir = dataset_dir / "BCCD" / "ImageSets" / "Main"
    split_ids: dict[str, list[str]] = {}

    for split_name in ("train", "val", "test"):
        split_path = split_dir / f"{split_name}.txt"
        image_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        split_ids[split_name] = [f"{image_id}.jpg" for image_id in image_ids]

    return split_ids


def apply_splits(counts_df: pd.DataFrame, split_ids: dict[str, list[str]]) -> pd.DataFrame:
    split_lookup = {
        filename: split_name for split_name, filenames in split_ids.items() for filename in filenames
    }
    result = counts_df.copy()
    result["split"] = result["filename"].map(split_lookup)
    missing = result["split"].isna()
    if missing.any():
        raise ValueError(f"Found files without split assignment: {result.loc[missing, 'filename'].tolist()[:5]}")
    return result


def compute_count_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = CLASSES,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch in metrics: true={y_true.shape}, pred={y_pred.shape}")

    metrics: dict[str, Any] = {"per_class": {}}

    for idx, class_name in enumerate(class_names):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        diff = pred_col - true_col
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))

        sse = float(np.sum(diff ** 2))
        centered = true_col - float(np.mean(true_col))
        sst = float(np.sum(centered ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else math.nan

        metrics["per_class"][class_name] = {
            "mae": mae,
            "rmse": float(math.sqrt(mse)),
            "r2": r2,
            "mean_true": float(np.mean(true_col)),
            "mean_pred": float(np.mean(pred_col)),
        }

    total_true = np.sum(y_true, axis=1)
    total_pred = np.sum(y_pred, axis=1)
    total_diff = total_pred - total_true
    total_mse = float(np.mean(total_diff ** 2))

    metrics["overall"] = {
        "mae_macro": float(np.mean([value["mae"] for value in metrics["per_class"].values()])),
        "rmse_macro": float(np.mean([value["rmse"] for value in metrics["per_class"].values()])),
        "total_count_mae": float(np.mean(np.abs(total_diff))),
        "total_count_rmse": float(math.sqrt(total_mse)),
        "exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }
    return metrics


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def prepare_yolo_dataset(
    dataset_dir: Path,
    output_dir: Path,
    split_ids: dict[str, list[str]],
) -> Path:
    bccd_dir = dataset_dir / "BCCD"
    image_dir = bccd_dir / "JPEGImages"
    annotation_dir = bccd_dir / "Annotations"

    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, filenames in split_ids.items():
        for filename in filenames:
            src_image = image_dir / filename
            dst_image = output_dir / "images" / split_name / filename
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            xml_path = annotation_dir / f"{Path(filename).stem}.xml"
            parsed = parse_voc_annotation(xml_path)
            width = parsed["image_width"]
            height = parsed["image_height"]

            label_lines = []
            for obj in parsed["objects"]:
                x_center = ((obj["xmin"] + obj["xmax"]) / 2.0) / width
                y_center = ((obj["ymin"] + obj["ymax"]) / 2.0) / height
                bbox_width = (obj["xmax"] - obj["xmin"]) / width
                bbox_height = (obj["ymax"] - obj["ymin"]) / height
                class_id = CLASS_TO_ID[obj["class_name"]]
                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                )

            label_path = output_dir / "labels" / split_name / f"{Path(filename).stem}.txt"
            label_path.write_text("\n".join(label_lines))

    yaml_path = output_dir / "bccd.yaml"
    yaml_payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: class_name for idx, class_name in enumerate(CLASSES)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))
    return yaml_path

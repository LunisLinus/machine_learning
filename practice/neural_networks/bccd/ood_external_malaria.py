from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bccd_utils import CLASSES, write_json  # noqa: E402
from ood_detection import (  # noqa: E402
    BCCDCorruptedDataset,
    build_variant_frame,
    collect_resnet_outputs,
    collect_yolo_outputs,
    evaluate_resnet_variant,
    fit_mahalanobis_reference,
    make_resnet_loader,
    mahalanobis_scores,
)
from train_count_regression import ResNetCountRegressor, decode_counts, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate realistic external OOD using the Malaria Cell Images dataset for ResNet50 and YOLOv8."
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=CURRENT_DIR / "artifacts" / "metadata" / "image_counts.csv",
    )
    parser.add_argument(
        "--malaria-dir",
        type=Path,
        default=CURRENT_DIR / "cell_images" / "cell_images",
        help="Path to extracted Malaria Cell Images dataset directory containing Parasitized/ and Uninfected/.",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=CURRENT_DIR / "artifacts" / "regression" / "multi_output" / "best_model.pt",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=CURRENT_DIR / "artifacts" / "detection" / "runs" / "bccd_yolo" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "artifacts" / "ood_external" / "malaria",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-class", type=int, default=120)
    parser.add_argument("--covariance-regularization", type=float, default=1e-3)
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.25)
    parser.add_argument("--yolo-iou-threshold", type=float, default=0.5)
    parser.add_argument("--yolo-embed-layers", type=int, nargs="+", default=[15, 18, 21])
    return parser.parse_args()


def build_malaria_frame(malaria_dir: Path, sample_per_class: int, seed: int) -> pd.DataFrame:
    class_dirs = {
        "Parasitized": malaria_dir / "Parasitized",
        "Uninfected": malaria_dir / "Uninfected",
    }

    records: list[dict[str, Any]] = []
    for label, class_dir in class_dirs.items():
        if not class_dir.exists():
            raise FileNotFoundError(f"Malaria subset not found: {class_dir}")

        image_paths = sorted([path for path in class_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        rng = np.random.default_rng(seed + len(records))
        if sample_per_class < len(image_paths):
            selected_idx = rng.choice(len(image_paths), size=sample_per_class, replace=False)
            image_paths = [image_paths[idx] for idx in sorted(selected_idx)]

        for image_path in image_paths:
            records.append(
                {
                    "filename": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "dataset_label": label,
                }
            )

    frame = pd.DataFrame(records).sort_values(["dataset_label", "filename"]).reset_index(drop=True)
    frame["split"] = "external"
    return frame


class ExternalImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), row["filename"], row["dataset_label"]


def make_external_loader(frame: pd.DataFrame, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = ExternalImageDataset(frame=frame, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_resnet_penultimate_features(model: ResNetCountRegressor, inputs: torch.Tensor) -> torch.Tensor:
    backbone = model.backbone
    x = backbone.conv1(inputs)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    return torch.flatten(x, 1)


def collect_external_resnet_outputs(
    model: ResNetCountRegressor,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()

    features: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    filenames: list[str] = []
    dataset_labels: list[str] = []

    with torch.no_grad():
        for images, batch_filenames, batch_labels in tqdm(loader, leave=False):
            images = images.to(device)
            batch_features = extract_resnet_penultimate_features(model, images)
            batch_predictions = model(images)

            features.append(batch_features.cpu().numpy())
            predictions.append(batch_predictions.cpu().numpy())
            filenames.extend(batch_filenames)
            dataset_labels.extend(batch_labels)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(predictions, axis=0),
        filenames,
        dataset_labels,
    )


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(len(id_scores), dtype=np.int32),
            np.ones(len(ood_scores), dtype=np.int32),
        ]
    )
    scores = np.concatenate([id_scores, ood_scores]).astype(np.float64)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    valid_indices = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[valid_indices[0]]) if len(valid_indices) > 0 else 1.0

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr95,
        "id_score_mean": float(np.mean(id_scores)),
        "id_score_std": float(np.std(id_scores)),
        "ood_score_mean": float(np.mean(ood_scores)),
        "ood_score_std": float(np.std(ood_scores)),
    }


def summarise_by_label(score_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label, part in score_df.groupby("dataset_label"):
        payload: dict[str, float] = {
            "count": float(len(part)),
            "ood_score_mean": float(part["ood_score"].mean()),
            "ood_score_std": float(part["ood_score"].std(ddof=0)),
        }
        if "pred_total_cells" in part.columns:
            payload["pred_total_cells_mean"] = float(part["pred_total_cells"].mean())
        if "num_detections" in part.columns:
            payload["num_detections_mean"] = float(part["num_detections"].mean())
            payload["mean_confidence_mean"] = float(part["mean_confidence"].mean())
        result[label] = payload
    return result


def run_resnet_external(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    malaria_frame: pd.DataFrame,
) -> dict[str, Any]:
    model = ResNetCountRegressor(num_outputs=len(CLASSES), pretrained=False)
    state_dict = torch.load(args.resnet_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    train_variant = build_variant_frame(train_frame, corruption_type="clean", severity=0.0, seed=args.seed)
    train_loader = make_resnet_loader(train_variant, args.image_size, args.batch_size, args.num_workers)
    train_features, _, _, _ = collect_resnet_outputs(model, train_loader, args.device)
    mean_vector, precision = fit_mahalanobis_reference(train_features, args.covariance_regularization)

    clean_variant = build_variant_frame(test_frame, corruption_type="clean", severity=0.0, seed=args.seed)
    clean_scores_df, _ = evaluate_resnet_variant(
        model=model,
        variant_frame=clean_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        mean_vector=mean_vector,
        precision=precision,
    )
    id_scores = clean_scores_df["ood_score"].to_numpy(dtype=np.float64)

    external_loader = make_external_loader(malaria_frame, args.image_size, args.batch_size, args.num_workers)
    features, predictions, filenames, dataset_labels = collect_external_resnet_outputs(model, external_loader, args.device)
    scores = mahalanobis_scores(features, mean_vector, precision)
    decoded_predictions = decode_counts(predictions)

    score_df = pd.DataFrame(
        {
            "filename": filenames,
            "dataset_label": dataset_labels,
            "ood_score": scores,
            "pred_RBC": decoded_predictions[:, 0],
            "pred_WBC": decoded_predictions[:, 1],
            "pred_Platelets": decoded_predictions[:, 2],
        }
    )
    score_df["pred_total_cells"] = score_df[["pred_RBC", "pred_WBC", "pred_Platelets"]].sum(axis=1)

    output_dir = args.output_dir / "resnet50"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(output_dir / "scores.csv", index=False)

    metrics = {
        "external_dataset": "Malaria Cell Images",
        "checkpoint": str(args.resnet_checkpoint.resolve()),
        "ood_metrics": compute_ood_metrics(id_scores=id_scores, ood_scores=score_df["ood_score"].to_numpy(dtype=np.float64)),
        "by_label": summarise_by_label(score_df),
        "sample_per_class": args.sample_per_class,
    }
    write_json(metrics, output_dir / "metrics.json")
    return {
        "model_name": "resnet50",
        "output_dir": str(output_dir.resolve()),
        "metrics_path": str((output_dir / "metrics.json").resolve()),
    }


def count_yolo_predictions(results: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_counts: list[list[int]] = []
    num_detections: list[int] = []
    mean_confidences: list[float] = []

    for result in results:
        counts = {class_name: 0 for class_name in CLASSES}
        if result.boxes is None or len(result.boxes) == 0:
            pred_counts.append([0, 0, 0])
            num_detections.append(0)
            mean_confidences.append(0.0)
            continue

        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        for class_id in class_ids:
            counts[CLASSES[class_id]] += 1

        pred_counts.append([counts[class_name] for class_name in CLASSES])
        num_detections.append(int(len(class_ids)))
        mean_confidences.append(float(np.mean(confidences)) if len(confidences) > 0 else 0.0)

    return (
        np.asarray(pred_counts, dtype=np.int32),
        np.asarray(num_detections, dtype=np.int32),
        np.asarray(mean_confidences, dtype=np.float32),
    )


def collect_external_yolo_outputs(
    feature_model: Any,
    detection_model: Any,
    malaria_frame: pd.DataFrame,
    batch_size: int,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    embed_layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    features: list[np.ndarray] = []
    pred_counts: list[np.ndarray] = []
    num_detections: list[np.ndarray] = []
    mean_confidences: list[np.ndarray] = []
    filenames: list[str] = []
    dataset_labels: list[str] = []

    for start_idx in tqdm(range(0, len(malaria_frame), batch_size), leave=False):
        batch = malaria_frame.iloc[start_idx : start_idx + batch_size]
        batch_images = [np.asarray(Image.open(path).convert("RGB")) for path in batch["image_path"]]
        batch_filenames = batch["filename"].tolist()
        batch_labels = batch["dataset_label"].tolist()

        batch_embeddings = feature_model.predict(
            source=batch_images,
            embed=embed_layers,
            batch=len(batch_images),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        batch_features = np.stack([tensor.detach().cpu().numpy() for tensor in batch_embeddings], axis=0)

        batch_results = detection_model.predict(
            source=batch_images,
            conf=conf_threshold,
            iou=iou_threshold,
            batch=len(batch_images),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        batch_pred_counts, batch_num_detections, batch_mean_confidences = count_yolo_predictions(batch_results)

        features.append(batch_features)
        pred_counts.append(batch_pred_counts)
        num_detections.append(batch_num_detections)
        mean_confidences.append(batch_mean_confidences)
        filenames.extend(batch_filenames)
        dataset_labels.extend(batch_labels)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(pred_counts, axis=0),
        np.concatenate(num_detections, axis=0),
        np.concatenate(mean_confidences, axis=0),
        filenames,
        dataset_labels,
    )


def run_yolo_external(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    malaria_frame: pd.DataFrame,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("The ultralytics package is required for YOLO external OOD evaluation.") from exc

    feature_model = YOLO(str(args.yolo_weights))
    detection_model = YOLO(str(args.yolo_weights))

    train_variant = build_variant_frame(train_frame, corruption_type="clean", severity=0.0, seed=args.seed)
    train_features, _, _, _, _, _ = collect_yolo_outputs(
        feature_model=feature_model,
        detection_model=detection_model,
        variant_frame=train_variant,
        batch_size=args.batch_size,
        imgsz=args.yolo_imgsz,
        device=args.device,
        conf_threshold=args.yolo_conf_threshold,
        iou_threshold=args.yolo_iou_threshold,
        embed_layers=args.yolo_embed_layers,
    )
    mean_vector, precision = fit_mahalanobis_reference(train_features, args.covariance_regularization)

    clean_variant = build_variant_frame(test_frame, corruption_type="clean", severity=0.0, seed=args.seed)
    clean_scores_df, _ = collect_clean_yolo_scores(
        feature_model=feature_model,
        detection_model=detection_model,
        variant_frame=clean_variant,
        args=args,
        mean_vector=mean_vector,
        precision=precision,
    )
    id_scores = clean_scores_df["ood_score"].to_numpy(dtype=np.float64)

    features, pred_counts, num_detections, mean_confidences, filenames, dataset_labels = collect_external_yolo_outputs(
        feature_model=feature_model,
        detection_model=detection_model,
        malaria_frame=malaria_frame,
        batch_size=args.batch_size,
        imgsz=args.yolo_imgsz,
        device=args.device,
        conf_threshold=args.yolo_conf_threshold,
        iou_threshold=args.yolo_iou_threshold,
        embed_layers=args.yolo_embed_layers,
    )
    scores = mahalanobis_scores(features, mean_vector, precision)

    score_df = pd.DataFrame(
        {
            "filename": filenames,
            "dataset_label": dataset_labels,
            "ood_score": scores,
            "pred_RBC": pred_counts[:, 0],
            "pred_WBC": pred_counts[:, 1],
            "pred_Platelets": pred_counts[:, 2],
            "num_detections": num_detections,
            "mean_confidence": mean_confidences,
        }
    )
    score_df["pred_total_cells"] = score_df[["pred_RBC", "pred_WBC", "pred_Platelets"]].sum(axis=1)

    output_dir = args.output_dir / "yolov8"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(output_dir / "scores.csv", index=False)

    metrics = {
        "external_dataset": "Malaria Cell Images",
        "weights": str(args.yolo_weights.resolve()),
        "ood_metrics": compute_ood_metrics(id_scores=id_scores, ood_scores=score_df["ood_score"].to_numpy(dtype=np.float64)),
        "by_label": summarise_by_label(score_df),
        "sample_per_class": args.sample_per_class,
        "embed_layers": args.yolo_embed_layers,
        "conf_threshold": args.yolo_conf_threshold,
        "iou_threshold": args.yolo_iou_threshold,
    }
    write_json(metrics, output_dir / "metrics.json")
    return {
        "model_name": "yolov8",
        "output_dir": str(output_dir.resolve()),
        "metrics_path": str((output_dir / "metrics.json").resolve()),
    }


def collect_clean_yolo_scores(
    feature_model: Any,
    detection_model: Any,
    variant_frame: pd.DataFrame,
    args: argparse.Namespace,
    mean_vector: np.ndarray,
    precision: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    features, pred_counts, targets, num_detections, mean_confidences, filenames = collect_yolo_outputs(
        feature_model=feature_model,
        detection_model=detection_model,
        variant_frame=variant_frame,
        batch_size=args.batch_size,
        imgsz=args.yolo_imgsz,
        device=args.device,
        conf_threshold=args.yolo_conf_threshold,
        iou_threshold=args.yolo_iou_threshold,
        embed_layers=args.yolo_embed_layers,
    )
    scores = mahalanobis_scores(features, mean_vector, precision)
    score_df = pd.DataFrame(
        {
            "filename": filenames,
            "ood_score": scores,
            "num_detections": num_detections,
            "mean_confidence": mean_confidences,
        }
    )
    return score_df, pred_counts


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.counts_csv).sort_values(["split", "filename"]).reset_index(drop=True)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    malaria_frame = build_malaria_frame(args.malaria_dir, sample_per_class=args.sample_per_class, seed=args.seed)

    runs = [
        run_resnet_external(args=args, train_frame=train_frame, test_frame=test_frame, malaria_frame=malaria_frame),
        run_yolo_external(args=args, train_frame=train_frame, test_frame=test_frame, malaria_frame=malaria_frame),
    ]

    summary = {
        "external_dataset": "Malaria Cell Images",
        "malaria_dir": str(args.malaria_dir.resolve()),
        "sample_per_class": args.sample_per_class,
        "output_dir": str(args.output_dir.resolve()),
        "runs": runs,
    }
    write_json(summary, args.output_dir / "summary.json")
    print(f"Malaria external OOD summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

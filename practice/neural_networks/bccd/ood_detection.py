from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from bccd_utils import CLASSES, compute_count_metrics, write_json
from train_count_regression import ResNetCountRegressor, decode_counts, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OOD detection for blurred and noisy BCCD images using trained ResNet50 and YOLOv8 models."
    )
    parser.add_argument(
        "--model-type",
        choices=("resnet50", "yolov8", "all"),
        default="all",
        help="Which model family to evaluate for OOD detection.",
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "metadata" / "image_counts.csv",
        help="Prepared CSV with image-level counts produced by prepare_bccd_data.py.",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "regression" / "multi_output" / "best_model.pt",
        help="Checkpoint of the trained ResNet50 regressor.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "detection" / "runs" / "bccd_yolo" / "weights" / "best.pt",
        help="Weights of the trained YOLOv8 detector.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "ood",
        help="Directory where OOD scores and metrics will be saved.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input size for ResNet50 OOD evaluation.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Input size for YOLOv8 OOD evaluation.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--blur-radii",
        type=float,
        nargs="+",
        default=[2.0, 4.0],
        help="Gaussian blur radii used to create OOD images.",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[15.0, 30.0],
        help="Standard deviations of Gaussian noise used to create OOD images in pixel space.",
    )
    parser.add_argument(
        "--covariance-regularization",
        type=float,
        default=1e-3,
        help="Diagonal regularization added to the covariance matrix for Mahalanobis scoring.",
    )
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.25)
    parser.add_argument("--yolo-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--yolo-embed-layers",
        type=int,
        nargs="+",
        default=[15, 18, 21],
        help="YOLOv8 layer indices used to extract pooled embeddings for Mahalanobis OOD detection.",
    )
    return parser.parse_args()


def apply_corruption(image: Image.Image, corruption_type: str, severity: float, rng: np.random.Generator) -> Image.Image:
    if corruption_type == "clean":
        return image
    if corruption_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if corruption_type == "noise":
        image_array = np.asarray(image, dtype=np.float32)
        noisy = image_array + rng.normal(0.0, severity, size=image_array.shape)
        noisy = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy)
    raise ValueError(f"Unsupported corruption_type={corruption_type}")


def make_variant_name(corruption_type: str, severity: float) -> str:
    if corruption_type == "clean":
        return "clean"
    prefix = "blur_radius" if corruption_type == "blur" else "noise_std"
    return f"{prefix}_{severity:g}"


def fit_mahalanobis_reference(train_features: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    mean_vector = np.mean(train_features, axis=0)
    covariance = np.cov(train_features, rowvar=False)
    covariance += regularization * np.eye(covariance.shape[0], dtype=np.float32)
    precision = np.linalg.pinv(covariance)
    return mean_vector, precision


def mahalanobis_scores(features: np.ndarray, mean_vector: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = features - mean_vector
    return np.einsum("bi,ij,bj->b", diff, precision, diff)


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


def build_variant_frame(frame: pd.DataFrame, corruption_type: str, severity: float, seed: int) -> pd.DataFrame:
    variant = frame.copy().reset_index(drop=True)
    variant["corruption_type"] = corruption_type
    variant["severity"] = severity
    variant["variant_name"] = make_variant_name(corruption_type, severity)
    variant["corruption_seed"] = [seed + idx for idx in range(len(variant))]
    return variant


def load_corrupted_image(row: pd.Series) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    rng = np.random.default_rng(int(row["corruption_seed"]))
    return apply_corruption(image, str(row["corruption_type"]), float(row["severity"]), rng)


class BCCDCorruptedDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = load_corrupted_image(row)
        target = torch.tensor(row.loc[list(CLASSES)].to_numpy(dtype=np.float32))
        return self.transform(image), target, row["filename"]


def make_resnet_loader(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = BCCDCorruptedDataset(frame=frame, image_size=image_size)
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


def collect_resnet_outputs(
    model: ResNetCountRegressor,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()

    features: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    filenames: list[str] = []

    with torch.no_grad():
        for images, batch_targets, batch_filenames in tqdm(loader, leave=False):
            images = images.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = extract_resnet_penultimate_features(model, images)
            batch_predictions = model(images)

            features.append(batch_features.cpu().numpy())
            predictions.append(batch_predictions.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
            filenames.extend(batch_filenames)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        filenames,
    )


def build_resnet_score_frame(
    variant_frame: pd.DataFrame,
    filenames: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    decoded_predictions = decode_counts(predictions)
    frame = pd.DataFrame(
        {
            "filename": filenames,
            "split": str(variant_frame["split"].iloc[0]),
            "corruption_type": str(variant_frame["corruption_type"].iloc[0]),
            "severity": float(variant_frame["severity"].iloc[0]),
            "variant_name": str(variant_frame["variant_name"].iloc[0]),
            "ood_score": scores,
        }
    )
    for idx, class_name in enumerate(CLASSES):
        frame[f"true_{class_name}"] = targets[:, idx].astype(np.int32)
        frame[f"pred_{class_name}"] = decoded_predictions[:, idx]
    frame["total_true"] = frame[[f"true_{class_name}" for class_name in CLASSES]].sum(axis=1)
    frame["total_pred"] = frame[[f"pred_{class_name}" for class_name in CLASSES]].sum(axis=1)
    frame["total_abs_error"] = (frame["total_true"] - frame["total_pred"]).abs()
    return frame


def evaluate_resnet_variant(
    model: ResNetCountRegressor,
    variant_frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    mean_vector: np.ndarray,
    precision: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    loader = make_resnet_loader(
        frame=variant_frame,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    features, predictions, targets, filenames = collect_resnet_outputs(model, loader, device)
    scores = mahalanobis_scores(features, mean_vector, precision)
    score_frame = build_resnet_score_frame(
        variant_frame=variant_frame,
        filenames=filenames,
        targets=targets,
        predictions=predictions,
        scores=scores,
    )
    count_metrics = compute_count_metrics(targets, decode_counts(predictions), class_names=CLASSES)
    return score_frame, count_metrics


def run_resnet_ood(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    model = ResNetCountRegressor(num_outputs=len(CLASSES), pretrained=False)
    state_dict = torch.load(args.resnet_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    train_variant = build_variant_frame(train_frame, corruption_type="clean", severity=0.0, seed=args.seed)
    train_loader = make_resnet_loader(
        frame=train_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_features, _, _, _ = collect_resnet_outputs(model, train_loader, args.device)
    mean_vector, precision = fit_mahalanobis_reference(train_features, args.covariance_regularization)

    variants: list[tuple[str, float]] = [("clean", 0.0)]
    variants.extend(("blur", radius) for radius in args.blur_radii)
    variants.extend(("noise", std) for std in args.noise_stds)

    score_frames: list[pd.DataFrame] = []
    metrics_payload: dict[str, Any] = {"variants": {}}

    clean_scores: np.ndarray | None = None
    clean_count_metrics: dict[str, Any] | None = None

    for corruption_type, severity in variants:
        variant_frame = build_variant_frame(test_frame, corruption_type=corruption_type, severity=severity, seed=args.seed)
        score_frame, count_metrics = evaluate_resnet_variant(
            model=model,
            variant_frame=variant_frame,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            mean_vector=mean_vector,
            precision=precision,
        )
        score_frames.append(score_frame)

        if corruption_type == "clean":
            clean_scores = score_frame["ood_score"].to_numpy(dtype=np.float64)
            clean_count_metrics = count_metrics
        else:
            metrics_payload["variants"][make_variant_name(corruption_type, severity)] = {
                "ood_metrics": compute_ood_metrics(clean_scores, score_frame["ood_score"].to_numpy(dtype=np.float64)),
                "count_metrics": count_metrics,
            }

    metrics_payload["clean_test_count_metrics"] = clean_count_metrics
    metrics_payload["checkpoint"] = str(args.resnet_checkpoint.resolve())
    metrics_payload["counts_csv"] = str(args.counts_csv.resolve())
    metrics_payload["covariance_regularization"] = args.covariance_regularization
    metrics_payload["blur_radii"] = args.blur_radii
    metrics_payload["noise_stds"] = args.noise_stds

    score_df = pd.concat(score_frames, ignore_index=True)
    score_df.to_csv(args.output_dir / "scores.csv", index=False)
    write_json(metrics_payload, args.output_dir / "metrics.json")

    resnet_dir = args.output_dir / "resnet50"
    resnet_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(resnet_dir / "scores.csv", index=False)
    write_json(metrics_payload, resnet_dir / "metrics.json")

    return {
        "model_name": "resnet50",
        "output_dir": str(resnet_dir.resolve()),
        "num_variants": len(metrics_payload["variants"]),
    }


def prepare_yolo_batch(
    variant_frame: pd.DataFrame,
    start_idx: int,
    batch_size: int,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    batch_images: list[np.ndarray] = []
    batch_targets: list[np.ndarray] = []
    batch_filenames: list[str] = []

    end_idx = min(start_idx + batch_size, len(variant_frame))
    for idx in range(start_idx, end_idx):
        row = variant_frame.iloc[idx]
        image = load_corrupted_image(row)
        batch_images.append(np.asarray(image))
        batch_targets.append(row.loc[list(CLASSES)].to_numpy(dtype=np.float32))
        batch_filenames.append(str(row["filename"]))

    return batch_images, np.stack(batch_targets, axis=0), batch_filenames


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


def collect_yolo_outputs(
    feature_model: Any,
    detection_model: Any,
    variant_frame: pd.DataFrame,
    batch_size: int,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    embed_layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    pred_counts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    num_detections: list[np.ndarray] = []
    mean_confidences: list[np.ndarray] = []
    filenames: list[str] = []

    for start_idx in tqdm(range(0, len(variant_frame), batch_size), leave=False):
        batch_images, batch_targets, batch_filenames = prepare_yolo_batch(variant_frame, start_idx, batch_size)

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
        targets.append(batch_targets)
        num_detections.append(batch_num_detections)
        mean_confidences.append(batch_mean_confidences)
        filenames.extend(batch_filenames)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(pred_counts, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(num_detections, axis=0),
        np.concatenate(mean_confidences, axis=0),
        filenames,
    )


def build_yolo_score_frame(
    variant_frame: pd.DataFrame,
    filenames: list[str],
    targets: np.ndarray,
    pred_counts: np.ndarray,
    scores: np.ndarray,
    num_detections: np.ndarray,
    mean_confidences: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "filename": filenames,
            "split": str(variant_frame["split"].iloc[0]),
            "corruption_type": str(variant_frame["corruption_type"].iloc[0]),
            "severity": float(variant_frame["severity"].iloc[0]),
            "variant_name": str(variant_frame["variant_name"].iloc[0]),
            "ood_score": scores,
            "num_detections": num_detections,
            "mean_confidence": mean_confidences,
        }
    )
    for idx, class_name in enumerate(CLASSES):
        frame[f"true_{class_name}"] = targets[:, idx].astype(np.int32)
        frame[f"pred_{class_name}"] = pred_counts[:, idx]
    frame["total_true"] = frame[[f"true_{class_name}" for class_name in CLASSES]].sum(axis=1)
    frame["total_pred"] = frame[[f"pred_{class_name}" for class_name in CLASSES]].sum(axis=1)
    frame["total_abs_error"] = (frame["total_true"] - frame["total_pred"]).abs()
    return frame


def evaluate_yolo_variant(
    feature_model: Any,
    detection_model: Any,
    variant_frame: pd.DataFrame,
    batch_size: int,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    embed_layers: list[int],
    mean_vector: np.ndarray,
    precision: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, pred_counts, targets, num_detections, mean_confidences, filenames = collect_yolo_outputs(
        feature_model=feature_model,
        detection_model=detection_model,
        variant_frame=variant_frame,
        batch_size=batch_size,
        imgsz=imgsz,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        embed_layers=embed_layers,
    )
    scores = mahalanobis_scores(features, mean_vector, precision)
    score_frame = build_yolo_score_frame(
        variant_frame=variant_frame,
        filenames=filenames,
        targets=targets,
        pred_counts=pred_counts,
        scores=scores,
        num_detections=num_detections,
        mean_confidences=mean_confidences,
    )
    count_metrics = compute_count_metrics(targets, pred_counts, class_names=CLASSES)
    return score_frame, count_metrics


def run_yolo_ood(
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("The ultralytics package is required for YOLOv8 OOD detection.") from exc

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

    variants: list[tuple[str, float]] = [("clean", 0.0)]
    variants.extend(("blur", radius) for radius in args.blur_radii)
    variants.extend(("noise", std) for std in args.noise_stds)

    score_frames: list[pd.DataFrame] = []
    metrics_payload: dict[str, Any] = {"variants": {}}

    clean_scores: np.ndarray | None = None
    clean_count_metrics: dict[str, Any] | None = None

    for corruption_type, severity in variants:
        variant_frame = build_variant_frame(test_frame, corruption_type=corruption_type, severity=severity, seed=args.seed)
        score_frame, count_metrics = evaluate_yolo_variant(
            feature_model=feature_model,
            detection_model=detection_model,
            variant_frame=variant_frame,
            batch_size=args.batch_size,
            imgsz=args.yolo_imgsz,
            device=args.device,
            conf_threshold=args.yolo_conf_threshold,
            iou_threshold=args.yolo_iou_threshold,
            embed_layers=args.yolo_embed_layers,
            mean_vector=mean_vector,
            precision=precision,
        )
        score_frames.append(score_frame)

        if corruption_type == "clean":
            clean_scores = score_frame["ood_score"].to_numpy(dtype=np.float64)
            clean_count_metrics = count_metrics
        else:
            metrics_payload["variants"][make_variant_name(corruption_type, severity)] = {
                "ood_metrics": compute_ood_metrics(clean_scores, score_frame["ood_score"].to_numpy(dtype=np.float64)),
                "count_metrics": count_metrics,
            }

    metrics_payload["clean_test_count_metrics"] = clean_count_metrics
    metrics_payload["weights"] = str(args.yolo_weights.resolve())
    metrics_payload["counts_csv"] = str(args.counts_csv.resolve())
    metrics_payload["covariance_regularization"] = args.covariance_regularization
    metrics_payload["blur_radii"] = args.blur_radii
    metrics_payload["noise_stds"] = args.noise_stds
    metrics_payload["embed_layers"] = args.yolo_embed_layers
    metrics_payload["conf_threshold"] = args.yolo_conf_threshold
    metrics_payload["iou_threshold"] = args.yolo_iou_threshold

    score_df = pd.concat(score_frames, ignore_index=True)
    yolo_dir = args.output_dir / "yolov8"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(yolo_dir / "scores.csv", index=False)
    write_json(metrics_payload, yolo_dir / "metrics.json")

    return {
        "model_name": "yolov8",
        "output_dir": str(yolo_dir.resolve()),
        "num_variants": len(metrics_payload["variants"]),
    }


def collect_existing_run_summaries(output_dir: Path) -> list[dict[str, Any]]:
    run_summaries: list[dict[str, Any]] = []

    resnet_metrics = output_dir / "resnet50" / "metrics.json"
    if resnet_metrics.exists():
        payload = {
            "model_name": "resnet50",
            "output_dir": str((output_dir / "resnet50").resolve()),
            "metrics_path": str(resnet_metrics.resolve()),
        }
        run_summaries.append(payload)

    yolo_metrics = output_dir / "yolov8" / "metrics.json"
    if yolo_metrics.exists():
        payload = {
            "model_name": "yolov8",
            "output_dir": str((output_dir / "yolov8").resolve()),
            "metrics_path": str(yolo_metrics.resolve()),
        }
        run_summaries.append(payload)

    root_resnet_metrics = output_dir / "metrics.json"
    if root_resnet_metrics.exists() and not any(run["model_name"] == "resnet50" for run in run_summaries):
        run_summaries.append(
            {
                "model_name": "resnet50",
                "output_dir": str(output_dir.resolve()),
                "metrics_path": str(root_resnet_metrics.resolve()),
            }
        )

    return run_summaries


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.counts_csv).sort_values(["split", "filename"]).reset_index(drop=True)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)

    if args.model_type in {"resnet50", "all"}:
        resnet_summary = run_resnet_ood(args=args, train_frame=train_frame, test_frame=test_frame)
        print(f"ResNet50 OOD metrics saved to: {Path(resnet_summary['output_dir']) / 'metrics.json'}")

    if args.model_type in {"yolov8", "all"}:
        yolo_summary = run_yolo_ood(args=args, train_frame=train_frame, test_frame=test_frame)
        print(f"YOLOv8 OOD metrics saved to: {Path(yolo_summary['output_dir']) / 'metrics.json'}")

    summary: dict[str, Any] = {
        "model_type": args.model_type,
        "output_dir": str(args.output_dir.resolve()),
        "runs": collect_existing_run_summaries(args.output_dir),
    }
    write_json(summary, args.output_dir / "summary.json")
    print(f"OOD summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

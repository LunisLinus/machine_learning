from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from bccd_utils import CLASSES, compute_count_metrics, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BCCDCountDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        target_classes: tuple[str, ...],
        train: bool,
        augmentation: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.target_classes = target_classes

        transform_steps: list[Any] = [
            transforms.Resize((image_size, image_size)),
        ]
        if train and augmentation != "none":
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
                ]
            )
            if augmentation == "strong":
                transform_steps.extend(
                    [
                        transforms.RandomAutocontrast(p=0.25),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.2),
                        transforms.RandomGrayscale(p=0.05),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                    ]
                )
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        targets = torch.tensor(row.loc[list(self.target_classes)].to_numpy(dtype=np.float32))
        return self.transform(image), targets, row["filename"]


class ResNetCountRegressor(nn.Module):
    def __init__(self, num_outputs: int, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_outputs = self.backbone(inputs)
        return torch.nn.functional.softplus(raw_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct count regression on BCCD images.")
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "metadata" / "image_counts.csv",
        help="Prepared CSV with image-level counts produced by prepare_bccd_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "regression",
        help="Directory where checkpoints, predictions and metrics will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--separate-models", action="store_true", help="Train one regressor per cell class.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization.")
    parser.add_argument(
        "--augmentation",
        choices=("none", "basic", "strong"),
        default="strong",
        help=(
            "Train-time image augmentation for count regression. "
            "The default keeps object counts unchanged while varying color, contrast, sharpness and blur."
        ),
    )
    return parser.parse_args()


def make_dataloaders(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    target_classes: tuple[str, ...],
    augmentation: str,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        split_frame = frame.loc[frame["split"] == split_name].reset_index(drop=True)
        dataset = BCCDCountDataset(
            frame=split_frame,
            image_size=image_size,
            target_classes=target_classes,
            train=split_name == "train",
            augmentation=augmentation,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    filenames: list[str] = []

    for images, batch_targets, batch_filenames in tqdm(loader, leave=False):
        images = images.to(device)
        batch_targets = batch_targets.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, batch_targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        predictions.append(outputs.detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())
        filenames.extend(batch_filenames)

    return (
        float(np.mean(losses)),
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        filenames,
    )


def decode_counts(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(predictions), a_min=0, a_max=None).astype(np.int32)


def train_single_configuration(
    frame: pd.DataFrame,
    target_classes: tuple[str, ...],
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaders = make_dataloaders(
        frame=frame,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_classes=target_classes,
        augmentation=args.augmentation,
    )

    model = ResNetCountRegressor(num_outputs=len(target_classes), pretrained=not args.no_pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(model, loaders["train"], criterion, args.device, optimizer)
        val_loss, val_pred, val_true, _ = run_epoch(model, loaders["val"], criterion, args.device, optimizer=None)

        val_metrics = compute_count_metrics(val_true, decode_counts(val_pred), class_names=target_classes)
        val_mae = val_metrics["overall"]["mae_macro"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae_macro": val_mae,
                "val_total_count_mae": val_metrics["overall"]["total_count_mae"],
            }
        )
        print(
            f"[{run_dir.name}] epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae_macro={val_mae:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    test_loss, test_pred, test_true, filenames = run_epoch(
        model, loaders["test"], criterion, args.device, optimizer=None
    )
    decoded_test_pred = decode_counts(test_pred)
    metrics = compute_count_metrics(test_true, decoded_test_pred, class_names=target_classes)
    metrics["test_loss"] = test_loss

    predictions_df = pd.DataFrame({"filename": filenames})
    for idx, class_name in enumerate(target_classes):
        predictions_df[f"true_{class_name}"] = test_true[:, idx].astype(int)
        predictions_df[f"pred_{class_name}"] = decoded_test_pred[:, idx]

    return predictions_df, metrics


def merge_single_target_predictions(prediction_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = prediction_frames[0]
    for frame in prediction_frames[1:]:
        merged = merged.merge(frame, on="filename", how="inner")
    return merged.sort_values("filename").reset_index(drop=True)


def collect_metrics_from_merged_predictions(predictions_df: pd.DataFrame) -> dict[str, Any]:
    true_matrix = predictions_df[[f"true_{class_name}" for class_name in CLASSES]].to_numpy(dtype=np.int32)
    pred_matrix = predictions_df[[f"pred_{class_name}" for class_name in CLASSES]].to_numpy(dtype=np.int32)
    return compute_count_metrics(true_matrix, pred_matrix, class_names=CLASSES)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    frame = pd.read_csv(args.counts_csv)
    frame = frame.sort_values(["split", "filename"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.separate_models:
        prediction_frames: list[pd.DataFrame] = []
        individual_metrics: dict[str, Any] = {}

        for class_name in CLASSES:
            run_dir = args.output_dir / f"single_{class_name.lower()}"
            predictions_df, metrics = train_single_configuration(
                frame=frame,
                target_classes=(class_name,),
                run_dir=run_dir,
                args=args,
            )
            prediction_frames.append(predictions_df)
            individual_metrics[class_name] = metrics

        merged_predictions = merge_single_target_predictions(prediction_frames)
        merged_metrics = collect_metrics_from_merged_predictions(merged_predictions)
        merged_predictions.to_csv(args.output_dir / "test_predictions_merged.csv", index=False)
        write_json(
            {
                "mode": "separate_models",
                "class_order": list(CLASSES),
                "augmentation": args.augmentation,
                "individual_metrics": individual_metrics,
                "merged_count_metrics": merged_metrics,
            },
            args.output_dir / "metrics.json",
        )
        print(f"Merged test predictions saved to: {args.output_dir / 'test_predictions_merged.csv'}")
    else:
        run_dir = args.output_dir / "multi_output"
        predictions_df, metrics = train_single_configuration(
            frame=frame,
            target_classes=CLASSES,
            run_dir=run_dir,
            args=args,
        )
        predictions_df.to_csv(args.output_dir / "test_predictions.csv", index=False)
        write_json(
            {
                "mode": "multi_output",
                "class_order": list(CLASSES),
                "augmentation": args.augmentation,
                "count_metrics": metrics,
            },
            args.output_dir / "metrics.json",
        )
        print(f"Test predictions saved to: {args.output_dir / 'test_predictions.csv'}")

    print(f"Regression metrics saved to: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

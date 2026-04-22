from __future__ import annotations

import argparse
from dataclasses import asdict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

from ..core.config import IMAGENET_MEAN, IMAGENET_STD, CarvanaConfig, prepare_runtime
from ..datasets.carvana_data import build_splits, create_loaders, load_mask, load_rgb
from ..modeling.segmentation import BCE_LOSS, DICE_LOSS, batch_iou_dice_from_logits, binary_scores, make_smp_unet


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    ious: list[float] = []
    dices: list[float] = []
    progress = tqdm(loader, leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)

        losses.append(loss.item())

        if is_train:
            if scaler is None:
                raise RuntimeError("GradScaler is required for training mode.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_description(f"train loss={np.mean(losses):.4f}")
        else:
            iou, dice = batch_iou_dice_from_logits(logits, masks, threshold=threshold)
            ious.append(iou)
            dices.append(dice)
            progress.set_description(f"valid loss={np.mean(losses):.4f} dice={np.mean(dices):.4f}")

    return {
        "loss": float(np.mean(losses)),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
    }


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    epochs: int,
    lr: float,
    experiment_name: str,
    report_dir,
    model_dir,
    threshold: float,
    validate_every: int,
) -> tuple[pd.DataFrame, str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: list[dict[str, float]] = []
    best_dice = -np.inf
    best_path = model_dir / f"{experiment_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            optimizer=optimizer,
            threshold=threshold,
            scaler=scaler,
        )

        if epoch % validate_every == 0 or epoch == epochs:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=use_amp,
                optimizer=None,
                threshold=threshold,
            )
            scheduler.step(val_metrics["dice"])
        else:
            val_metrics = {"loss": np.nan, "iou": np.nan, "dice": np.nan}

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }
        history.append(row)

        print(
            f"[{experiment_name}] epoch {epoch:02d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val dice={val_metrics['dice'] if not np.isnan(val_metrics['dice']) else 'skip'}"
        )

        if not np.isnan(val_metrics["dice"]) and val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(report_dir / f"{experiment_name}_history.csv", index=False)
    return history_df, str(best_path)


@torch.inference_mode()
def predict_prob_map_unet(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    transform = A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    transformed = transform(image=image_rgb)
    x = transformed["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last)
    logits = model(x)
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


@torch.inference_mode()
def evaluate_unet_on_dataframe(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="U-Net evaluation"):
        image = load_rgb(row.image_path)
        true_mask = (load_mask(row.mask_path) > 127).astype(np.uint8)
        prob_small = predict_prob_map_unet(model, image, image_size, device)
        prob_full = cv2.resize(
            prob_small,
            (true_mask.shape[1], true_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        iou, dice = binary_scores(prob_full, true_mask)
        rows.append({"id": row.id, "iou": iou, "dice": dice})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Carvana segmentation models outside Jupyter.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--debug-max-samples", type=int, default=None)
    parser.add_argument("--skip-bce", action="store_true")
    parser.add_argument("--skip-dice", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CarvanaConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.lr is not None:
        config.lr = args.lr
    if args.encoder is not None:
        config.encoder_name = args.encoder
    if args.img_size is not None:
        config.img_size = tuple(args.img_size)
    if args.validate_every is not None:
        config.validate_every = args.validate_every
    if args.debug_max_samples is not None:
        config.debug_max_samples = args.debug_max_samples

    device, use_amp = prepare_runtime(config)
    print("device:", device)
    print("config:", asdict(config))

    _, train_df, val_df = build_splits(config)
    train_loader, val_loader = create_loaders(train_df, val_df, config)
    print(f"train: {len(train_df)} images")
    print(f"val: {len(val_df)} images")

    histories: dict[str, pd.DataFrame] = {}
    best_paths: dict[str, str] = {}

    if not args.skip_bce:
        bce_model = make_smp_unet(config.encoder_name).to(device).to(memory_format=torch.channels_last)
        bce_history, bce_best_path = train_model(
            model=bce_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=BCE_LOSS,
            device=device,
            use_amp=use_amp,
            epochs=config.epochs,
            lr=config.lr,
            experiment_name="smp_unet_bce",
            report_dir=config.report_dir,
            model_dir=config.model_dir,
            threshold=config.threshold,
            validate_every=config.validate_every,
        )
        histories["BCE"] = bce_history
        best_paths["BCE"] = bce_best_path

    if not args.skip_dice:
        dice_model = make_smp_unet(config.encoder_name).to(device).to(memory_format=torch.channels_last)
        dice_history, dice_best_path = train_model(
            model=dice_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=DICE_LOSS,
            device=device,
            use_amp=use_amp,
            epochs=config.epochs,
            lr=config.lr,
            experiment_name="smp_unet_dice",
            report_dir=config.report_dir,
            model_dir=config.model_dir,
            threshold=config.threshold,
            validate_every=config.validate_every,
        )
        histories["DiceLoss"] = dice_history
        best_paths["DiceLoss"] = dice_best_path

    if histories:
        combined = []
        for name, history in histories.items():
            hist = history.copy()
            hist["model"] = name
            combined.append(hist)
        pd.concat(combined, ignore_index=True).to_csv(config.report_dir / "combined_history.csv", index=False)

    if args.skip_eval or not best_paths:
        return

    summary_rows = []
    for name, best_path in best_paths.items():
        model = make_smp_unet(config.encoder_name).to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(best_path, map_location=device))
        eval_df = evaluate_unet_on_dataframe(model, val_df, config.img_size, device)
        eval_df.to_csv(config.report_dir / f"{name.lower()}_eval.csv", index=False)
        summary_rows.append(
            {
                "model": name,
                "val_iou_mean": eval_df["iou"].mean(),
                "val_dice_mean": eval_df["dice"].mean(),
                "checkpoint": best_path,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("val_dice_mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(config.report_dir / "unet_summary.csv", index=False)
    print(summary_df)

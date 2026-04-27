from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torchvision import models, transforms
except ImportError:
    models = None
    transforms = None


CLASSES = ["RBC", "WBC", "Platelets"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
BCCD_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
BASE_DIR = Path(__file__).resolve().parent
HTTP_HEADERS = {
    "User-Agent": "bccd-cell-counting/0.1 (educational notebook; contact: local)",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}
REALISTIC_OOD_SOURCES = [
    {
        "name": "ecoli_niaid_16578744517.jpg",
        "title": "File:E. coli Bacteria (16578744517).jpg",
        "source": "Wikimedia Commons / NIAID, E. coli bacteria microscopy",
    },
    {
        "name": "ecoli_electron_microscopy.jpg",
        "title": "File:Escherichia coli electron microscopy.jpg",
        "source": "Wikimedia Commons / Janice Haney Carr, E. coli SEM",
    },
    {
        "name": "mouse_tissue_histology_23180949464.jpg",
        "title": "File:Mouse tissue, stained histology preparation (23180949464).jpg",
        "source": "Wikimedia Commons / ZEISS Microscopy, mouse tissue histology",
    },
    {
        "name": "mouse_tissue_histology_23782972906.jpg",
        "title": "File:Mouse tissue, stained histology preparation (23782972906).jpg",
        "source": "Wikimedia Commons / ZEISS Microscopy, mouse tissue histology",
    },
]


def require_torchvision() -> None:
    if models is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for the ResNet50 pipeline. Install dependencies with: "
            "pip install -e practice/neural_networks/bccd"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def download_file(url: str, dst: Path, chunk_size: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, headers=HTTP_HEADERS) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dst.open("wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))


def ensure_bccd(data_dir: Path, force_download: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = find_bccd_root(data_dir)
    if existing and not force_download:
        return existing

    archive = data_dir / Path(urlparse(BCCD_URL).path).name
    if force_download or not archive.exists():
        print(f"Downloading BCCD from {BCCD_URL}")
        download_file(BCCD_URL, archive)

    extract_dir = data_dir / "raw"
    if force_download and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    root = find_bccd_root(data_dir)
    if not root:
        raise FileNotFoundError("Cannot find BCCD/JPEGImages and BCCD/Annotations after extraction.")
    return root


def find_bccd_root(data_dir: Path) -> Path | None:
    for candidate in [data_dir, *data_dir.glob("**/BCCD")]:
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    return None


def parse_annotation(xml_path: Path, images_dir: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    if not filename:
        filename = f"{xml_path.stem}.jpg"

    image_path = images_dir / filename
    if not image_path.exists():
        matches = list(images_dir.glob(f"{xml_path.stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Image for {xml_path.name} was not found.")
        image_path = matches[0]

    size = root.find("size")
    width = int(size.findtext("width")) if size is not None else Image.open(image_path).width
    height = int(size.findtext("height")) if size is not None else Image.open(image_path).height
    boxes: list[dict] = []
    counts = {name: 0 for name in CLASSES}
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        boxes.append({"class": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        counts[name] += 1

    return {
        "image_id": xml_path.stem,
        "image_path": str(image_path),
        "xml_path": str(xml_path),
        "width": width,
        "height": height,
        "boxes": boxes,
        **counts,
        "total_cells": sum(counts.values()),
    }


def build_dataframe(bccd_root: Path, out_csv: Path) -> pd.DataFrame:
    annotations_dir = bccd_root / "Annotations"
    images_dir = bccd_root / "JPEGImages"
    rows = [parse_annotation(path, images_dir) for path in sorted(annotations_dir.glob("*.xml"))]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_for_csv = df.copy()
    df_for_csv["boxes"] = df_for_csv["boxes"].apply(json.dumps)
    df_for_csv.to_csv(out_csv, index=False)
    return df


def draw_boxes(image_path: str, boxes: list[dict]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"RBC": "red", "WBC": "lime", "Platelets": "cyan"}
    for box in boxes:
        cls = box["class"]
        xy = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        draw.rectangle(xy, outline=colors[cls], width=3)
        draw.text((box["xmin"], max(0, box["ymin"] - 12)), cls, fill=colors[cls])
    return image


def analyze_and_visualize(df: pd.DataFrame, figures_dir: Path, sample_count: int = 6) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("\nDataset summary")
    print(df[CLASSES + ["total_cells", "width", "height"]].describe().round(2))
    print("\nClass totals")
    print(df[CLASSES].sum().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[CLASSES].sum().plot(kind="bar", ax=axes[0], color=["#d94b4b", "#4b9bd9", "#48a868"])
    axes[0].set_title("Cells by class")
    axes[0].set_ylabel("count")
    df[CLASSES].plot(kind="hist", bins=20, alpha=0.6, ax=axes[1])
    axes[1].set_title("Per-image count distributions")
    axes[1].set_xlabel("cells per image")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    sample_df = df.sample(min(sample_count, len(df)), random_state=42)
    cols = min(3, len(sample_df))
    rows = math.ceil(len(sample_df) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        ax.imshow(draw_boxes(row.image_path, row.boxes))
        ax.set_title(f"{row.image_id}: RBC={row.RBC}, WBC={row.WBC}, Platelets={row.Platelets}")
        ax.axis("off")
    for ax in axes[len(sample_df):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "annotated_samples.png", dpi=160)
    plt.close(fig)


def split_dataframe(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = pd.qcut(df["total_cells"], q=min(5, df["total_cells"].nunique()), duplicates="drop")
    if stratify.value_counts().min() < 2:
        stratify = None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class BCCDCountDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        train: bool,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):
        require_torchvision()
        self.df = df.reset_index(drop=True)
        self.target_mean = target_mean
        self.target_std = target_std
        aug = [
            transforms.Resize((image_size, image_size)),
        ]
        if train:
            aug += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            ]
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row.image_path).convert("RGB")
        target_np = np.array([row[c] for c in CLASSES], dtype=np.float32)
        if self.target_mean is not None and self.target_std is not None:
            target_np = (target_np - self.target_mean) / self.target_std
        target = torch.tensor(target_np, dtype=torch.float32)
        return self.transform(image), target


class CountRegressor(nn.Module):
    def __init__(self, pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        require_torchvision()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(CLASSES)),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@torch.no_grad()
def predict_counts(
    model: CountRegressor,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for images, y in loader:
        images = images.to(device)
        pred = model(images).cpu().numpy()
        y_np = y.numpy()
        if target_mean is not None and target_std is not None:
            pred = pred * target_std + target_mean
            y_np = y_np * target_std + target_mean
        pred = np.clip(pred, 0, None)
        preds.append(pred)
        targets.append(y_np)
    return np.vstack(preds), np.vstack(targets)


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {}
    rounded = np.rint(np.clip(y_pred, 0, None))
    for idx, cls in enumerate(CLASSES):
        result[cls] = {
            "mae": float(mean_absolute_error(y_true[:, idx], rounded[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], rounded[:, idx]))),
            "r2": float(r2_score(y_true[:, idx], rounded[:, idx])),
        }
    result["macro_mae"] = float(mean_absolute_error(y_true, rounded))
    result["macro_rmse"] = float(np.sqrt(mean_squared_error(y_true, rounded)))
    result["exact_image_match"] = float((rounded == y_true).all(axis=1).mean())
    return result


def train_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    work_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    pretrained: bool,
    freeze_backbone: bool,
    num_workers: int,
) -> tuple[CountRegressor, dict]:
    target_mean = train_df[CLASSES].to_numpy(dtype=np.float32).mean(axis=0)
    target_std = train_df[CLASSES].to_numpy(dtype=np.float32).std(axis=0) + 1e-6
    train_ds = BCCDCountDataset(train_df, image_size=image_size, train=True, target_mean=target_mean, target_std=target_std)
    val_ds = BCCDCountDataset(val_df, image_size=image_size, train=False, target_mean=target_mean, target_std=target_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    model = CountRegressor(pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_mae = float("inf")
    best_path = work_dir / "regression_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"regression epoch {epoch}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        preds, y_true = predict_counts(model, val_loader, device, target_mean=target_mean, target_std=target_std)
        metrics = count_metrics(y_true, preds)
        val_mae = metrics["macro_mae"]
        print(f"epoch={epoch} train_loss={train_loss / len(train_ds):.4f} val_macro_mae={val_mae:.3f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    preds, y_true = predict_counts(model, val_loader, device, target_mean=target_mean, target_std=target_std)
    return model, count_metrics(y_true, preds)


def yolo_box_line(box: dict, width: float, height: float) -> str:
    x_center = ((box["xmin"] + box["xmax"]) / 2) / width
    y_center = ((box["ymin"] + box["ymax"]) / 2) / height
    box_width = (box["xmax"] - box["xmin"]) / width
    box_height = (box["ymax"] - box["ymin"]) / height
    return f"{CLASS_TO_ID[box['class']]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def prepare_yolo_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, split_df in [("train", train_df), ("val", val_df)]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for _, row in split_df.iterrows():
            src = Path(row.image_path)
            dst = out_dir / "images" / split / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            label_path = out_dir / "labels" / split / f"{src.stem}.txt"
            lines = [yolo_box_line(box, row.width, row.height) for box in row.boxes]
            label_path.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out_dir / "bccd.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def train_and_eval_yolo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    work_dir: Path,
    device: torch.device,
    yolo_model: str,
    yolo_epochs: int,
    image_size: int,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics or run with --skip-yolo.") from exc

    yaml_path = prepare_yolo_dataset(train_df, val_df, work_dir / "yolo_dataset")
    yolo_device = 0 if device.type == "cuda" else "cpu"
    model = YOLO(yolo_model)
    model.train(
        data=str(yaml_path),
        epochs=yolo_epochs,
        imgsz=image_size,
        batch=-1,
        patience=5,
        project=str(work_dir / "yolo_runs"),
        name=f"bccd_{Path(yolo_model).stem}",
        exist_ok=True,
        device=yolo_device,
        verbose=False,
    )

    image_paths = [str(path) for path in (work_dir / "yolo_dataset" / "images" / "val").glob("*")]
    raw_predictions = []
    for result in model.predict(image_paths, imgsz=image_size, conf=0.001, iou=0.5, device=yolo_device, verbose=False):
        if result.boxes is None or result.boxes.cls is None:
            raw_predictions.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
            continue
        raw_predictions.append(
            (
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy().astype(np.float32),
            )
        )

    y_true = val_df[CLASSES].to_numpy(dtype=np.float32)
    conf_grid = np.linspace(0.05, 0.70, 14)
    best_conf = 0.25
    best_mae = float("inf")
    best_pred: np.ndarray | None = None
    for conf in conf_grid:
        predictions = []
        for classes, scores in raw_predictions:
            counts = np.zeros(len(CLASSES), dtype=np.float32)
            keep = scores >= conf
            for cls_id in classes[keep]:
                if 0 <= cls_id < len(CLASSES):
                    counts[cls_id] += 1
            predictions.append(counts)
        y_pred = np.vstack(predictions)
        macro_mae = float(mean_absolute_error(y_true, y_pred))
        if macro_mae < best_mae:
            best_mae = macro_mae
            best_conf = float(conf)
            best_pred = y_pred

    assert best_pred is not None
    metrics = count_metrics(y_true, best_pred)
    metrics["selected_confidence"] = best_conf
    return metrics


@torch.no_grad()
def extract_features(model: CountRegressor, image_paths: Iterable[str], image_size: int, device: torch.device, batch_size: int = 32) -> np.ndarray:
    require_torchvision()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    images = []
    all_features = []
    model.eval()
    for path in image_paths:
        images.append(transform(Image.open(path).convert("RGB")))
        if len(images) == batch_size:
            batch = torch.stack(images).to(device)
            all_features.append(model.features(batch).cpu().numpy())
            images.clear()
    if images:
        batch = torch.stack(images).to(device)
        all_features.append(model.features(batch).cpu().numpy())
    return np.vstack(all_features)


def save_corrupted_images(df: pd.DataFrame, out_dir: Path, corruption: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in df.iterrows():
        image = Image.open(row.image_path).convert("RGB")
        if corruption == "blur":
            corrupted = image.filter(ImageFilter.GaussianBlur(radius=5))
        elif corruption == "noise":
            image_np = np.asarray(image).astype(np.float32)
            noise = np.random.normal(0, 35, image_np.shape).astype(np.float32)
            corrupted_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
            corrupted = Image.fromarray(corrupted_np)
        else:
            raise ValueError(corruption)
        dst = out_dir / f"{Path(row.image_path).stem}_{corruption}.jpg"
        corrupted.save(dst)
        paths.append(str(dst))
    return paths


def list_external_images(external_dir: Path | None, fallback_dir: Path, count: int) -> list[str]:
    paths: list[Path] = []
    if external_dir and external_dir.exists():
        paths = [p for p in external_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and is_valid_image(p)]
    if paths:
        return [str(p) for p in paths[:count]]

    if external_dir:
        paths = download_realistic_external_ood(external_dir)
        if paths:
            return [str(p) for p in paths[:count]]

    fallback_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    rng = np.random.default_rng(42)
    for idx in range(count):
        base = rng.normal(120, 45, size=(360, 480, 3)).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        for _ in range(120):
            center_x, center_y = rng.integers([0, 0], [480, 360])
            axes = rng.integers([8, 3], [42, 16])
            color = tuple(int(x) for x in rng.integers(40, 230, size=3))
            bbox = [
                int(center_x - axes[0]),
                int(center_y - axes[1]),
                int(center_x + axes[0]),
                int(center_y + axes[1]),
            ]
            draw.ellipse(bbox, fill=color)
        dst = fallback_dir / f"synthetic_non_blood_{idx:03d}.jpg"
        image.save(dst)
        generated.append(str(dst))
    return generated


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def resolve_commons_image_url(title: str, width: int = 800) -> str:
    response = request_with_retries(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": width,
            "titles": title,
        },
        headers={
            **HTTP_HEADERS,
            "Accept": "application/json",
        },
        stream=False,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        imageinfo = page.get("imageinfo") or []
        if imageinfo:
            return imageinfo[0].get("thumburl") or imageinfo[0]["url"]
    raise RuntimeError(f"Commons image URL was not resolved for {title!r}.")


def request_with_retries(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    stream: bool = False,
    attempts: int = 4,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers or HTTP_HEADERS,
                stream=stream,
                timeout=60,
                allow_redirects=True,
            )
            if response.status_code == 429 and attempt < attempts - 1:
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 3.0 * (attempt + 1)
                response.close()
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def download_image(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with request_with_retries(url, headers=HTTP_HEADERS, stream=True) as response:
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type.lower():
            raise RuntimeError(f"Expected image content from {url}, got content-type={content_type!r}.")
        total = int(response.headers.get("content-length", 0))
        with dst.open("wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))


def download_realistic_external_ood(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    metadata = []
    for item in REALISTIC_OOD_SOURCES:
        dst = out_dir / item["name"]
        if dst.exists() and not is_valid_image(dst):
            dst.unlink()
        if not dst.exists():
            try:
                url = resolve_commons_image_url(item["title"])
                download_image(url, dst)
                with Image.open(dst) as image:
                    image.verify()
            except Exception as exc:
                print(f"Could not download external OOD image {item['name']}: {exc}")
                if dst.exists():
                    dst.unlink()
                continue
        else:
            url = str(dst)
        downloaded.append(dst)
        metadata.append({**item, "resolved_url": url})
        time.sleep(1.0)

    if metadata:
        (out_dir / "sources.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return downloaded


def image_quality_features(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((224, 224))
    gray = np.asarray(image, dtype=np.float32) / 255.0
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    return np.array(
        [
            float(laplacian.var()),
            float(np.mean(np.abs(dx))),
            float(np.mean(np.abs(dy))),
            float(gray.std()),
        ],
        dtype=np.float32,
    )


def extract_quality_features(image_paths: Iterable[str]) -> np.ndarray:
    return np.vstack([image_quality_features(path) for path in image_paths])


def fit_ood_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def ood_scores(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.mean(np.square((features - mean) / std), axis=1)


def combined_ood_scores(
    deep_features: np.ndarray,
    quality_features: np.ndarray,
    deep_mean: np.ndarray,
    deep_std: np.ndarray,
    quality_mean: np.ndarray,
    quality_std: np.ndarray,
) -> np.ndarray:
    deep_score = ood_scores(deep_features, deep_mean, deep_std)
    quality_score = ood_scores(quality_features, quality_mean, quality_std)
    return deep_score + quality_score


def run_ood_detection(
    model: CountRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    work_dir: Path,
    image_size: int,
    device: torch.device,
    external_ood_dir: Path | None,
) -> dict:
    train_features = extract_features(model, train_df.image_path, image_size, device)
    train_quality = extract_quality_features(train_df.image_path)
    mean, std = fit_ood_statistics(train_features)
    quality_mean, quality_std = fit_ood_statistics(train_quality)
    train_scores = combined_ood_scores(train_features, train_quality, mean, std, quality_mean, quality_std)
    threshold = float(np.quantile(train_scores, 0.95))

    clean_features = extract_features(model, val_df.image_path, image_size, device)
    clean_quality = extract_quality_features(val_df.image_path)
    clean_scores = combined_ood_scores(clean_features, clean_quality, mean, std, quality_mean, quality_std)

    blur_paths = save_corrupted_images(val_df, work_dir / "ood" / "blur", "blur")
    noise_paths = save_corrupted_images(val_df, work_dir / "ood" / "noise", "noise")
    external_paths = list_external_images(external_ood_dir, work_dir / "ood" / "synthetic_external", len(val_df))

    groups = {"clean": clean_scores}
    for name, paths in [("blur", blur_paths), ("noise", noise_paths), ("external", external_paths)]:
        feats = extract_features(model, paths, image_size, device)
        quality = extract_quality_features(paths)
        groups[name] = combined_ood_scores(feats, quality, mean, std, quality_mean, quality_std)

    metrics = {"threshold_train_p95": threshold}
    for name, scores in groups.items():
        metrics[name] = {
            "mean_score": float(scores.mean()),
            "ood_rate": float((scores > threshold).mean()),
        }

    for name in ["blur", "noise", "external"]:
        y_true = np.r_[np.zeros_like(groups["clean"]), np.ones_like(groups[name])]
        y_score = np.r_[groups["clean"], groups[name]]
        metrics[name]["roc_auc_vs_clean"] = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, scores in groups.items():
        ax.hist(scores, bins=24, alpha=0.45, density=True, label=name)
    ax.axvline(threshold, color="black", linestyle="--", label="train p95 threshold")
    ax.set_title("OOD scores from ResNet50 feature distance")
    ax.set_xlabel("mean squared z-score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(work_dir / "figures" / "ood_scores.png", dpi=160)
    plt.close(fig)
    return metrics


def save_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics to {path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCCD regression, YOLO counting, and OOD detection")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--work-dir", type=Path, default=BASE_DIR / "bccd_work")
    parser.add_argument("--external-ood-dir", type=Path, default=BASE_DIR / "data" / "external_ood")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet ResNet50 weights. Disable with --no-pretrained for fully offline runs.",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="Ultralytics detector checkpoint, e.g. yolo26n.pt or yolov8n.pt.")
    parser.add_argument("--yolo-epochs", type=int, default=10)
    parser.add_argument("--yolo-img-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "figures").mkdir(parents=True, exist_ok=True)

    bccd_root = ensure_bccd(args.data_dir, force_download=args.force_download)
    df = build_dataframe(bccd_root, args.work_dir / "bccd_dataset.csv")
    analyze_and_visualize(df, args.work_dir / "figures")
    train_df, val_df = split_dataframe(df, args.seed)

    model, regression_metrics = train_regressor(
        train_df=train_df,
        val_df=val_df,
        work_dir=args.work_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.img_size,
        lr=args.lr,
        pretrained=args.pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        num_workers=args.num_workers,
    )

    metrics = {"regression_resnet50": regression_metrics}
    if not args.skip_yolo:
        metrics[f"detection_{Path(args.yolo_model).stem}_counting"] = train_and_eval_yolo(
            train_df=train_df,
            val_df=val_df,
            work_dir=args.work_dir,
            device=device,
            yolo_model=args.yolo_model,
            yolo_epochs=args.yolo_epochs,
            image_size=args.yolo_img_size,
        )
    else:
        metrics[f"detection_{Path(args.yolo_model).stem}_counting"] = "skipped"

    metrics["ood_detection"] = run_ood_detection(
        model=model,
        train_df=train_df,
        val_df=val_df,
        work_dir=args.work_dir,
        image_size=args.img_size,
        device=device,
        external_ood_dir=args.external_ood_dir,
    )
    save_metrics(metrics, args.work_dir / "metrics.json")


if __name__ == "__main__":
    main()

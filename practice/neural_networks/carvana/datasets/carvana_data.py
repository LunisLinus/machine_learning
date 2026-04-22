from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import requests
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..core.config import IMAGENET_MEAN, IMAGENET_STD, IMG_EXTS, MASK_EXTS, CarvanaConfig


def resolve_yandex_download_url(public_key: str) -> str:
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    response = requests.get(base_url + urlencode({"public_key": public_key}), timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "href" not in payload:
        raise RuntimeError(f"Yandex Disk did not return direct download link: {payload}")
    return payload["href"]


def download_file(url: str, dst_path: Path, chunk_size: int = 2**20) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dst_path, "wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=dst_path.name) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))
    return dst_path


def extract_zip(archive_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(dst_dir)


def list_files_by_ext(root: Path, exts: set[str]) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]


def has_enough_pairs(root: Path) -> bool:
    image_files = [path for path in list_files_by_ext(root, IMG_EXTS) if "_mask" not in path.stem.lower()]
    mask_files = [
        path for path in list_files_by_ext(root, MASK_EXTS)
        if "mask" in path.stem.lower() or "mask" in str(path.parent).lower()
    ]
    return len(image_files) > 100 and len(mask_files) > 100


def extract_relevant_inner_archives(root: Path) -> None:
    if has_enough_pairs(root):
        return

    inner_archives = sorted(root.rglob("*.zip"))
    if not inner_archives:
        return

    def archive_score(path: Path) -> int:
        name = path.name.lower()
        score = 0
        if "train" in name:
            score += 3
        if "mask" in name:
            score += 3
        if "test" in name:
            score -= 5
        if "sample" in name or "submission" in name:
            score -= 3
        return score

    selected = sorted((path for path in inner_archives if archive_score(path) > 0), key=archive_score, reverse=True)
    for archive_path in selected:
        dst = archive_path.with_suffix("")
        if dst.exists() and any(dst.iterdir()):
            continue
        extract_zip(archive_path, dst)


def ensure_dataset(config: CarvanaConfig) -> None:
    outer_archive = config.raw_dir / "carvana-image-masking-challenge.zip"
    if not outer_archive.exists():
        download_url = resolve_yandex_download_url(config.public_yandex_url)
        download_file(download_url, outer_archive)

    if not any(config.extract_dir.iterdir()):
        extract_zip(outer_archive, config.extract_dir)

    extract_relevant_inner_archives(config.extract_dir)


def canonical_image_key(path: Path) -> str:
    return path.stem


def canonical_mask_key(path: Path) -> str:
    return re.sub(r"_mask$", "", path.stem)


def rank_image_path(path: Path) -> tuple[int, int, int]:
    path_str = str(path).lower()
    return (
        int("train_hq" in path_str),
        int("/train/" in path_str or "\\train\\" in path_str or "train" in path.parent.name.lower()),
        path.stat().st_size if path.exists() else 0,
    )


def rank_mask_path(path: Path) -> tuple[int, int, int]:
    path_str = str(path).lower()
    return (
        int("train_masks" in path_str),
        int("mask" in path_str),
        path.stat().st_size if path.exists() else 0,
    )


def discover_image_mask_pairs(root: Path) -> pd.DataFrame:
    image_candidates: dict[str, list[Path]] = {}
    for path in list_files_by_ext(root, IMG_EXTS):
        if path.name.startswith(".") or "_mask" in path.stem.lower():
            continue
        image_candidates.setdefault(canonical_image_key(path), []).append(path)

    mask_candidates: dict[str, list[Path]] = {}
    for path in list_files_by_ext(root, MASK_EXTS):
        if path.name.startswith("."):
            continue
        if "mask" not in path.stem.lower() and "mask" not in str(path.parent).lower():
            continue
        mask_candidates.setdefault(canonical_mask_key(path), []).append(path)

    image_map = {key: sorted(paths, key=rank_image_path, reverse=True)[0] for key, paths in image_candidates.items()}
    mask_map = {key: sorted(paths, key=rank_mask_path, reverse=True)[0] for key, paths in mask_candidates.items()}

    common_ids = sorted(set(image_map) & set(mask_map))
    if not common_ids:
        raise RuntimeError("Could not find image-mask pairs after extraction.")

    rows = []
    for sample_id in common_ids:
        parts = sample_id.split("_")
        rows.append(
            {
                "id": sample_id,
                "car_id": parts[0],
                "angle_id": parts[1] if len(parts) > 1 else None,
                "image_path": str(image_map[sample_id]),
                "mask_path": str(mask_map[sample_id]),
            }
        )

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def maybe_limit_dataframe(frame: pd.DataFrame, debug_max_samples: int | None, seed: int) -> pd.DataFrame:
    if debug_max_samples is None:
        return frame

    unique_cars = frame["car_id"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    selected_cars: list[str] = []
    selected_count = 0
    for car_id in unique_cars:
        car_rows = frame[frame["car_id"] == car_id]
        if selected_count + len(car_rows) > debug_max_samples and selected_cars:
            break
        selected_cars.append(car_id)
        selected_count += len(car_rows)

    return frame[frame["car_id"].isin(selected_cars)].reset_index(drop=True)


def build_splits(config: CarvanaConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset(config)
    frame = discover_image_mask_pairs(config.extract_dir)
    frame = maybe_limit_dataframe(frame, config.debug_max_samples, config.seed)

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.seed)
    train_idx, val_idx = next(splitter.split(frame, groups=frame["car_id"]))
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    return frame, train_df, val_df


def build_transforms(img_size: tuple[int, int]) -> tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    valid_transform = A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return train_transform, valid_transform


class CarvanaSegDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transforms: A.Compose | None = None):
        self.image_paths = frame["image_path"].astype(str).tolist()
        self.mask_paths = frame["mask_path"].astype(str).tolist()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])
        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)

        return image, mask.float()


def create_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: CarvanaConfig,
) -> tuple[DataLoader, DataLoader]:
    train_transform, valid_transform = build_transforms(config.img_size)
    train_ds = CarvanaSegDataset(train_df, transforms=train_transform)
    val_ds = CarvanaSegDataset(val_df, transforms=valid_transform)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask

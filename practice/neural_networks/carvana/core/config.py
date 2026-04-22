from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class CarvanaConfig:
    public_yandex_url: str = "https://disk.yandex.ru/d/v3KwZ5UJ8Uoshw"
    workdir: Path = Path(__file__).resolve().parents[1] / "carvana_work"
    seed: int = 42
    val_size: float = 0.2
    debug_max_samples: int | None = 2048
    num_workers: int = min(8, os.cpu_count() or 2)
    img_size: tuple[int, int] = (256, 256)
    batch_size: int = 32 if torch.cuda.is_available() else 8
    epochs: int = 10
    lr: float = 1e-3
    encoder_name: str = "mobilenet_v2"
    threshold: float = 0.5
    validate_every: int = 5
    max_train_steps: int | None = 64
    max_val_steps: int | None = 16

    @property
    def raw_dir(self) -> Path:
        return self.workdir / "raw"

    @property
    def extract_dir(self) -> Path:
        return self.workdir / "extracted"

    @property
    def model_dir(self) -> Path:
        return self.workdir / "models"

    @property
    def report_dir(self) -> Path:
        return self.workdir / "reports"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def prepare_runtime(config: CarvanaConfig) -> tuple[torch.device, bool]:
    for path in [
        config.workdir,
        config.raw_dir,
        config.extract_dir,
        config.model_dir,
        config.report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    cv2.setNumThreads(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    return device, use_amp

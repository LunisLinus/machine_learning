from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_histories(histories: dict[str, pd.DataFrame], metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False

    for name, history in histories.items():
        if metric not in history.columns:
            continue
        plot_df = history[["epoch", metric]].dropna().copy()
        if plot_df.empty:
            continue
        ax.plot(plot_df["epoch"], plot_df[metric], marker="o", linewidth=2, label=name)
        plotted = True

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Carvana training histories from CSV files.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories = {path.stem: pd.read_csv(path) for path in args.csv_files}
    for metric in ["train_loss", "val_dice", "val_iou", "val_loss"]:
        plot_histories(histories, metric)

#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv nicht gefunden unter: {csv_path}")
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def plot_training(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = df.columns

    # 1) Train Loss
    plt.figure()
    plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_losses.png", dpi=200)

    # 2) Val Loss
    if "val/box_loss" in cols:
        plt.figure()
        plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
        plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
        plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Validation Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "val_losses.png", dpi=200)

    # 3) Metrics
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    available = [m for m in metric_keys if m in cols]
    if available:
        plt.figure()
        for m in available:
            plt.plot(df["epoch"], df[m], label=m)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "val_metrics.png", dpi=200)

    # 4) Learning Rates
    lr_keys = [k for k in cols if k.startswith("lr/pg")]
    if lr_keys:
        plt.figure()
        for k in lr_keys:
            plt.plot(df["epoch"], df[k], label=k)
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rates per Parameter Group")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "lrs.png", dpi=200)

    print(f"[OK] Plots gespeichert unter: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualisiere Ultralytics YOLO Trainings-Logs (results.csv)."
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Pfad zu results.csv. "
            "Standard: ../runs/detect/train_yolov8s_gpu/results.csv relativ zu diesem Skript."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # = Forschsem\SegemtationTrainig_v1

    if args.results:
        csv_path = Path(args.results).expanduser().resolve()
        out_dir = csv_path.parent
    else:
        csv_path = project_root / "runs" / "detect" / "train_yolov8s_gpu" / "results.csv"
        out_dir = csv_path.parent

    df = load_results(csv_path)
    plot_training(df, out_dir)


if __name__ == "__main__":
    main()

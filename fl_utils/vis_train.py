
"""
训练过程可视化与CSV记录工具

功能：
- 每轮将 epoch/test_loss/test_acc/bkd_loss/bkd_acc/lr 追加写入 CSV
- 以 epoch 为横坐标，将 loss/acc/bkd_loss/bkd_acc 画到一张图的 4 个子图中
- 每次训练自动新建保存目录：saved/训练过程可视化/时间 + name
  其中 name 的格式对齐 main/clean.py 中 wandb 的 name（attack_type_dataset_model_agg_method_poison_start_epoch）
"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """兼容 wandb.config(属性访问) / dict(键访问) 两种配置对象。"""
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg.get(key, default)
    return default


def build_run_name_like_clean_py(cfg: Any) -> str:
    """
    对齐 main/clean.py (50-54) 的 name 格式：
    attack_type_dataset_model_agg_method_poison_start_epoch
    """
    attack_type = _get_cfg_value(cfg, "attack_type", "unknown_attack")
    dataset = _get_cfg_value(cfg, "dataset", "unknown_dataset")
    model = _get_cfg_value(cfg, "model", "unknown_model")
    agg_method = _get_cfg_value(cfg, "agg_method", "unknown_agg")
    poison_start_epoch = _get_cfg_value(cfg, "poison_start_epoch", "unknown_poison_start")
    return f"{attack_type}_{dataset}_{model}_{agg_method}_{poison_start_epoch}"


def _now_timestamp() -> str:
    # 文件夹名更友好：20260108-153012
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_root_dir() -> str:
    # fl_utils/vis_train.py -> 项目根目录 -> saved/训练过程可视化
    here = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(proj_root, "saved", "train_process_visualization")


@dataclass
class TrainLogRow:
    epoch: int
    test_loss: float
    test_acc: float
    bkd_loss: float
    bkd_acc: float
    lr: float


class TrainProcessVisualizer:
    """
    训练过程记录与可视化器：
    - 初始化时创建独立 run 文件夹
    - log() 每次追加一行 CSV，并更新图像文件（覆盖写）
    """

    def __init__(self, cfg: Any, root_dir: Optional[str] = None, enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled

        self.run_name = build_run_name_like_clean_py(cfg)
        self.timestamp = _now_timestamp()

        base_dir = root_dir or _default_root_dir()
        self.run_dir = os.path.join(base_dir, f"{self.timestamp}_{self.run_name}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "train_log.csv")
        self.fig_path_png = os.path.join(self.run_dir, "train_curves.png")
        self.fig_path_pdf = os.path.join(self.run_dir, "train_curves.pdf")

        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "test_loss", "test_acc", "bkd_loss", "bkd_acc", "lr"],
            )
            writer.writeheader()

    def log(self, epoch: int, test_loss: float, test_acc: float, bkd_loss: float, bkd_acc: float, lr: float) -> None:
        if not self.enabled:
            return

        row = TrainLogRow(
            epoch=int(epoch),
            test_loss=float(test_loss),
            test_acc=float(test_acc),
            bkd_loss=float(bkd_loss),
            bkd_acc=float(bkd_acc),
            lr=float(lr),
        )
        self._append_row(row)
        # 覆盖更新曲线图（简单可靠，代价可接受）
        try:
            self._plot_from_csv()
        except Exception as e:
            # 可视化失败不影响训练
            print(f"[TrainProcessVisualizer] 绘图失败：{e}")

    def _append_row(self, row: TrainLogRow) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "test_loss", "test_acc", "bkd_loss", "bkd_acc", "lr"],
            )
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "test_loss": row.test_loss,
                    "test_acc": row.test_acc,
                    "bkd_loss": row.bkd_loss,
                    "bkd_acc": row.bkd_acc,
                    "lr": row.lr,
                }
            )

    def _read_rows(self) -> List[TrainLogRow]:
        rows: List[TrainLogRow] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r is None:
                    continue
                try:
                    rows.append(
                        TrainLogRow(
                            epoch=int(float(r["epoch"])),
                            test_loss=float(r["test_loss"]),
                            test_acc=float(r["test_acc"]),
                            bkd_loss=float(r["bkd_loss"]),
                            bkd_acc=float(r["bkd_acc"]),
                            lr=float(r["lr"]),
                        )
                    )
                except Exception:
                    # 跳过坏行
                    continue
        rows.sort(key=lambda x: x.epoch)
        return rows

    def _plot_from_csv(self) -> None:
        # headless 环境
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self._read_rows()
        if len(rows) == 0:
            return

        epochs = [r.epoch for r in rows]
        test_loss = [r.test_loss for r in rows]
        test_acc = [r.test_acc for r in rows]
        bkd_loss = [r.bkd_loss for r in rows]
        bkd_acc = [r.bkd_acc for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        fig.suptitle(f"训练过程可视化：{self.run_name}")

        ax = axes[0, 0]
        ax.plot(epochs, test_loss, label="test_loss", color="tab:blue")
        ax.set_title("Test Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs, test_acc, label="test_acc", color="tab:green")
        ax.set_title("Test Acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("acc (%)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, bkd_loss, label="bkd_loss", color="tab:orange")
        ax.set_title("Backdoor Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(epochs, bkd_acc, label="bkd_acc", color="tab:red")
        ax.set_title("Backdoor Acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("acc (%)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(self.fig_path_png, bbox_inches="tight")
        fig.savefig(self.fig_path_pdf, bbox_inches="tight")
        plt.close(fig)



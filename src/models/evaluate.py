# src/models/evaluate.py

"""
evaluate.py
Запуск:
    python -m src.models.evaluate --weights /path/to/model.pt
Если --weights не указан → берётся самый новый runs/**/weights/best.pt
"""

from __future__ import annotations
import argparse, glob
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from src.utils.metrics import summarise_metrics
from src.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


def newest_best() -> Path | None:
    """Ищет самый свежий *best.pt* в runs/*/weights/."""
    pattern = str(CFG.paths.runs / "**" / "weights" / "best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    # сортируем по времени модификации
    newest = max(candidates, key=lambda p: Path(p).stat().st_mtime)
    ts = datetime.fromtimestamp(Path(newest).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log.info(f"Автоматически выбран {newest} (mtime {ts})")
    return Path(newest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, help=".pt файл с весами")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights or newest_best()
    if not weights or not weights.exists():
        raise FileNotFoundError("Не удалось определить файл весов; передайте его через --weights")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(Path(__file__).parent / "dataset.yaml"),
        imgsz=CFG.train.img_size,
        device=CFG.train.device,
        batch=CFG.train.batch,
        verbose=True,
    )
    summary = summarise_metrics(metrics)
    log.info(f"⭐ Результаты:\n{summary}")


if __name__ == "__main__":
    main()

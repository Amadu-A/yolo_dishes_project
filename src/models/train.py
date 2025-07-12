# src/models/train.py

""""
train.py
Обучение YOLO-11 (Ultralytics ≥ 8.3.165)

Запуск:
    python -m src.models.train --size s|n|x [--epochs 50] \
                               [--img 640] [--batch 16] [--lr0 0.01] \
                               [--weights /path/to/yolo11s.pt]
"""

from __future__ import annotations
import argparse, inspect
from pathlib import Path
from typing import Optional

from ultralytics import YOLO
import ultralytics
from src.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────── CLI ────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", default=CFG.train.model_size,
                   choices=("n", "s", "x"),
                   help="Размер модели: n|s|x")
    p.add_argument("--epochs", type=int, default=CFG.train.epochs)
    p.add_argument("--img", type=int, default=CFG.train.img_size,
                   help="Размер входного изображения (квадрат)")
    p.add_argument("--batch", type=int, default=CFG.train.batch)
    p.add_argument("--lr0", type=float, default=CFG.train.lr0,
                   help="Начальный learning-rate")
    p.add_argument("--weights", type=Path,
                   help="Явный путь к .pt или .yaml")
    return p.parse_args()


# ─────────────── helpers ──────────────
def _yaml_in_pkg(name: str) -> Optional[Path]:
    root = Path(inspect.getfile(ultralytics)).parent
    hits = list(root.rglob(f"{name}.yaml"))
    return hits[0] if hits else None


def _load(model_path: str | Path) -> YOLO:
    try:
        return YOLO(str(model_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл «{model_path}» не найден.") from e


def get_model(size: str, explicit: Optional[Path]) -> YOLO:
    base = f"yolo11{size}"
    if explicit:
        logger.info(f"Использую указанный файл: {explicit}")
        return _load(explicit)

    # ① официальный .pt
    try:
        logger.info(f"Пробую скачать/загрузить {base}.pt")
        return YOLO(f"{base}.pt")
    except FileNotFoundError:
        logger.info("  ↳ веса недоступны")

    # ② YAML в пакете
    if (yml := _yaml_in_pkg(base)):
        logger.info(f"Нашёл YAML в пакете: {yml}")
        return YOLO(str(yml))

    raise RuntimeError(
        f"Не найдено ни {base}.pt, ни YAML. "
        "Обновите ultralytics или передайте --weights."
    )


# ──────────────── main ────────────────
def main():
    args = parse_args()

    import torch
    if torch.cuda.is_available():
        logger.info(f"🟢 CUDA: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("🚫 CUDA недоступна — обучение на CPU")

    model = get_model(args.size, args.weights)

    logger.info(
        f"▶️  Старт обучения YOLO-11-{args.size} "
        f"({args.img}px, batch {args.batch}) на {args.epochs} эпох"
    )

    results = model.train(
        data=str(Path(__file__).parent / "dataset.yaml"),
        imgsz=args.img,
        epochs=args.epochs,
        batch=args.batch,
        device=CFG.train.device,
        lr0=args.lr0,
        project=CFG.paths.runs,
        name=f"exp11_{args.size}",
        exist_ok=True,
    )
    logger.info(results)


if __name__ == "__main__":
    main()

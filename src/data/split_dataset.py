# src/data/split_dataset.py

"""
split_dataset.py
Разбивает полный набор изображений/разметки на train/val/test в пропорции 70/20/10.
Запуск: python -m src.data.split_dataset
"""

from pathlib import Path
import random
import shutil
from src.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)
random.seed(42)

def split(images: list[Path], train=0.7, val=0.2):
    """
    Делит список путей по пропорциям.
    :returns: кортеж (train_paths, val_paths, test_paths)
    """
    n = len(images)
    random.shuffle(images)
    n_train = int(n * train)
    n_val = int(n * val)
    return images[:n_train], images[n_train:n_train+n_val], images[n_train+n_val:]

def copy_pairs(pairs: list[Path], dst_img_dir: Path, dst_lbl_dir: Path):
    """
    Копирует изображения вместе с их .txt разметкой.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in pairs:
        shutil.copy(img_path, dst_img_dir / img_path.name)
        lbl = img_path.with_suffix(".txt")
        shutil.copy(lbl, dst_lbl_dir / lbl.name)

def run():
    imgs = list((CFG.paths.data_processed).glob("*.jpg"))
    tr, vl, ts = split(imgs)
    logger.info(f"Split: {len(tr)} train, {len(vl)} val, {len(ts)} test")
    for subset, paths in zip(("train", "val", "test"), (tr, vl, ts)):
        copy_pairs(
            paths,
            CFG.paths.dataset / subset / "images",
            CFG.paths.dataset / subset / "labels"
        )

if __name__ == "__main__":
    run()

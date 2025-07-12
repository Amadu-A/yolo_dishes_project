# src/data/extract_frames.py

"""
augment.py
Применяет набор аугментаций Albumentations к каждому изображению+разметке.
Запуск: python -m src.data.augment
"""

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
from src.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- описываем пайплайн Albumentations ----------------------------------------
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),                 # случайное отражение
        A.RandomBrightnessContrast(0.2, 0.2),    # изменение яркости и контраста
        A.Blur(blur_limit=3, p=0.2),             # лёгкое размытие
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.ColorJitter(p=0.3)                     # сдвиг цветового тона
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)

def augment_image(img_path: Path, label_path: Path, dst_img: Path, dst_lbl: Path) -> None:
    """
    Аугментирует одну пару (image, label).
    YOLO-разметка: x_center y_center w h class_id
    """
    image = cv2.imread(str(img_path))
    # читаем yolo-разметку
    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.split())
            bboxes.append((x, y, w, h))
            class_labels.append(int(cls))

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_img = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_labels = augmented["class_labels"]

    cv2.imwrite(str(dst_img), aug_img)
    # сохраняем bbox обратно в YOLO-формате
    with open(dst_lbl, "w") as f:
        for (x, y, w, h), cls in zip(aug_bboxes, aug_labels):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def run():
    src_imgs = list((CFG.paths.frames).glob("*.jpg"))
    logger.info(f"Augmenting {len(src_imgs)} images")
    for img in tqdm(src_imgs):
        label = CFG.paths.labels / (img.stem + ".txt")
        if not label.exists():
            logger.warning(f"No label for {img.name}; skipping")
            continue
        for idx in range(3):  # создаём 3 аугментированных копии
            dst_img = CFG.paths.data_processed / f"{img.stem}_aug{idx}.jpg"
            dst_lbl = CFG.paths.data_processed / f"{img.stem}_aug{idx}.txt"
            augment_image(img, label, dst_img, dst_lbl)

if __name__ == "__main__":
    CFG.paths.data_processed.mkdir(parents=True, exist_ok=True)
    run()

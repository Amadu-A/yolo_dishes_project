"""
convert_cvat_xml.py
Преобразует файл annotations.xml (CVAT 1.1) в набор
YOLO-текстовиков по одному файлу на кадр.

Запуск:
    python -m src.data.convert_cvat_xml \
           --xml data/raw/cvat_backups/annotations.xml
Создаст *.txt в data/raw/labels/
"""

import argparse, xml.etree.ElementTree as ET
from pathlib import Path
from src.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Список классов должен 1-в-1 совпадать с dataset.yaml --------------
CLASSES = ["dish", "cup", "fork", "knife", "spoon", "teapot", "basket"]
CLASS2ID = {name: idx for idx, name in enumerate(CLASSES)}

def polygon_to_bbox(points: str):
    """
    CVAT polygon → YOLO bbox (x_center, y_center, w, h в относительных координатах)
    :param points: строка "x1,y1;x2,y2;..."
    """
    xs, ys = [], []
    for p in points.split(";"):
        x, y = map(float, p.split(","))
        xs.append(x); ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)

def box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h):
    """Координаты из пикселей → нормированные (0-1)."""
    xc = (xtl + xbr) / 2 / img_w
    yc = (ytl + ybr) / 2 / img_h
    w  = (xbr - xtl) / img_w
    h  = (ybr - ytl) / img_h
    return xc, yc, w, h

def convert(xml_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.iter("image"):
        fname = image.attrib["name"]          # frame_000000.jpg
        img_w = float(image.attrib["width"])
        img_h = float(image.attrib["height"])
        label_lines = []

        # --- boxes --------------------------------------------------------
        for box in image.iter("box"):
            lbl = box.attrib["label"]
            if lbl not in CLASS2ID:
                logger.warning(f"Unknown label {lbl} – пропускаем")
                continue
            xtl = float(box.attrib["xtl"]); ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"]); ybr = float(box.attrib["ybr"])
            xc, yc, w, h = box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h)
            label_lines.append(f"{CLASS2ID[lbl]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # --- polygons → bbox ---------------------------------------------
        for poly in image.iter("polygon"):
            lbl = poly.attrib["label"]
            if lbl not in CLASS2ID:
                logger.warning(f"Unknown label {lbl} – пропускаем")
                continue
            xtl, ytl, xbr, ybr = polygon_to_bbox(poly.attrib["points"])
            xc, yc, w, h = box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h)
            label_lines.append(f"{CLASS2ID[lbl]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # --- сохраняем YOLO-txt ------------------------------------------
        out_txt = out_dir / Path(fname).with_suffix(".txt").name
        with open(out_txt, "w") as f:
            f.write("\n".join(label_lines))
        logger.info(f"{out_txt.name}: {len(label_lines)} objects")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True,
                    help="Путь к annotations.xml")
    args = ap.parse_args()
    convert(args.xml, CFG.paths.labels)
    logger.info("Conversion done")

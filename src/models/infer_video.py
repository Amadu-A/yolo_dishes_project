# src/models/infer_video.py
"""
Прогоняет видео через обученную YOLO-11 и сохраняет новый ролик с бокcами.
Пример:
    python -m src.models.infer_video \
        --video data/raw/video.mp4 \
        --weights runs/exp11_s/weights/best.pt \
        --out results/out_video.mp4 \
        --img 960 --conf 0.25
"""

from pathlib import Path
import argparse, cv2, math, tqdm
from ultralytics import YOLO

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video",  type=Path, required=True, help="Исходное видео")
    p.add_argument("--weights", type=Path, required=True, help="Файл .pt")
    p.add_argument("--out",   type=Path, default="out.mp4", help="Куда сохранить результат")
    p.add_argument("--img",   type=int,  default=640, help="Размер стороны кадра для инференса")
    p.add_argument("--conf",  type=float, default=0.25, help="Порог confidence")
    p.add_argument("--batch", type=int,  default=16,   help="Batch-size для инференса")
    return p.parse_args()

def main():
    args = parse()
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    batch, frames_idx = [], []
    for idx in tqdm.trange(total, desc="Inference"):
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        frames_idx.append(idx)

        if len(batch) == args.batch or idx == total - 1:
            results = model(batch, imgsz=args.img, conf=args.conf, verbose=False)
            for img_i, res in enumerate(results):
                plotted = res.plot()          # возвращает BGR-кадр с боксами
                writer.write(plotted)
            batch, frames_idx = [], []

    cap.release()
    writer.release()
    print(f"✅ Saved → {args.out.resolve()}")

if __name__ == "__main__":
    main()

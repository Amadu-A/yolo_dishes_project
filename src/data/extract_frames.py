# src/data/extract_frames.py

"""
extract_frames.py
Извлекает равномерные кадры из видео-файла и сохраняет их в data/raw/frames.
Запуск:  python -m src.data.extract_frames
"""

import cv2                                                       # OpenCV – работа с видео
from pathlib import Path
from src.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)

VIDEO_EXTS = (".mp4", ".mov", ".mkv")

def extract(video_path: Path, dst_dir: Path, fps: int) -> None:
    """
    :param video_path: путь к видеофайлу .mp4
    :param dst_dir: куда сохранять кадры
    :param fps: сколько кадров в секунду сохраняем
    """
    logger.info(f"Extracting frames from {video_path}")
    cap = cv2.VideoCapture(str(video_path))                      # открываем видео
    orig_fps = cap.get(cv2.CAP_PROP_FPS)                         # исходная частота кадров видео
    step = int(round(orig_fps / fps))                            # шаг по кадрам
    count = 0                                                    # порядковый номер кадра
    saved = 0                                                    # счётчик сохранённых изображений
    dst_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()                                   # читаем кадр
        if not ok:                                               # конец файла
            break
        if count % step == 0:                                    # сохраняем каждый step-й кадр
            fname = dst_dir / f"frame_{saved:06d}{CFG.extract.img_suffix}"
            cv2.imwrite(str(fname), frame)                       # сохраняем кадр
            saved += 1
        count += 1

    cap.release()
    logger.info(f"Done: {saved} frames saved to {dst_dir}")

if __name__ == "__main__":
    video = CFG.paths.data_raw / "video.mp4"                     # предполагаемое имя файла
    if not video.exists():
        candidates = [p for p in CFG.paths.data_raw.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        if len(candidates) == 1:
            video = candidates[0]
            logger.info(f"Auto-detected video: {video.name}")
        else:
            raise FileNotFoundError(
                f"Не найден файл video.mp4 и не удалось однозначно выбрать видео из {VIDEO_EXTS}"
            )

    extract(video, CFG.paths.frames, CFG.extract.fps)

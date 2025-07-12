# src/main.py

"""
main.py
Упрощённый CLI-оркестратор. Один скрипт – весь пайплайн.

Пример:
    python -m src.main all --video /path/to/video.mp4
"""

import argparse, shutil
from pathlib import Path
from src.config import CFG
from src.utils.logger import get_logger
from src.data.extract_frames import extract
from src.data.augment import run as augment_run
from src.data.split_dataset import run as split_run
from src.models.train import main as train_main
from src.models.evaluate import main as eval_main
from src.report.make_report import main as report_main

logger = get_logger(__name__)

def parse():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--video", type=Path, required=True)

    return p.parse_args()

def run_all(video: Path):
    # 1) копируем видео в data/raw
    dst = CFG.paths.data_raw / "video.mp4"
    shutil.copy(video, dst)
    # 2) кадры
    extract(dst, CFG.paths.frames, CFG.extract.fps)
    # 3) аугментация и split
    augment_run()
    split_run()
    # 4) обучения трёх размеров моделей
    for size in ("n", "s", "x"):
        train_main(["--size", size, "--epochs", str(CFG.train.epochs)])
    # 5) оценка каждой best.pt
    for best in CFG.paths.runs.rglob("best.pt"):
        eval_main(["--weights", str(best)])
    # 6) отчёт
    report_main()

def main():
    args = parse()
    if args.cmd == "all":
        run_all(args.video)

if __name__ == "__main__":
    main()

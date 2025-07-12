# src/main.py
"""
main.py
Упрощённый CLI-оркестратор: один скрипт — весь пайплайн YOLO-11
(от «сырых» данных до итогового отчёта).

Примеры:
    # полный цикл
    python -m src.main all --video /path/to/video.mp4

    # обучение одной модели
    python -m src.main train --size s --epochs 50

    # инференс «по-короткому» (возьмёт последнее video.* и последний best.pt)
    python -m src.main infer
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from src.config import CFG
from src.utils.logger import get_logger

# ─── единичные этапы ──────────────────────────────────────────────────── #
from src.data.extract_frames import extract
from src.data.augment import run as augment_run
from src.data.split_dataset import run as split_run
from src.models.train import main as train_main
from src.models.evaluate import main as eval_main
from src.models.infer_video import main as infer_video_main
from src.report.make_report import main as report_main

logger = get_logger(__name__)

# ─────────────────────── helpers ─────────────────────────────────────── #
def _dispatch(func, argv: List[str] | None = None) -> None:
    """Запускает подскрипт *func* с подменённым argv (чтобы его argparse отработал)."""
    sys.argv = [func.__module__.split(".")[-1]] + (argv or [])
    func()


def _most_recent(patterns: list[str]) -> Optional[Path]:
    """Возвращает самый «свежий» файл из списка glob-паттернов (или None)."""
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(Path(CFG.paths.root).glob(pat))
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _default_video() -> Path:
    vid = _most_recent(
        [
            str(CFG.paths.data_raw / "*.mp4"),
            str(CFG.paths.data_raw / "*.mov"),
            str(CFG.paths.data_raw / "*.MOV"),
        ]
    )
    if not vid:
        raise FileNotFoundError("В папке data/raw не найдено ни одного видео (*.mp4/ *.MOV)")
    return vid


def _default_weights() -> Path:
    w = _most_recent([str(CFG.paths.runs / "**/best.pt")])
    if not w:
        raise FileNotFoundError("Не найдено ни одного best.pt в папке runs/")
    return w


# ──────────────────────────── CLI ────────────────────────────────────── #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO-11 dish-detection pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # полный цикл
    all_cmd = sub.add_parser("all", help="полный пайплайн")
    all_cmd.add_argument("--video", type=Path, required=True, help="исходный ролик *.mp4/ *.MOV")

    # обучение
    tr = sub.add_parser("train", help="обучить модель (n/s/x)")
    tr.add_argument("--size", default=CFG.train.model_size, choices=("n", "s", "x"))
    tr.add_argument("--epochs", type=int, default=CFG.train.epochs)

    # оценка
    ev = sub.add_parser("eval", help="оценить веса")
    ev.add_argument("--weights", type=Path, required=True)

    # инференс (все аргументы -> опциональны)
    inf = sub.add_parser("infer", help="инференс нового видео")
    inf.add_argument("--video", type=Path, help="видео (по умолч. — последнее в data/raw)")
    inf.add_argument("--weights", type=Path, help="веса (по умолч. — самый свежий best.pt)")
    inf.add_argument("--out", type=Path, help="файл вывода (.mp4). "
                     "Если не указан — results/<video>_boxes.mp4")
    inf.add_argument("--img", type=int, default=640)
    inf.add_argument("--conf", type=float, default=0.25)
    inf.add_argument("--batch", type=int, default=16)

    # отчёт
    sub.add_parser("report", help="собрать Markdown-отчёт")

    return p


# ─────────────────────── большой цикл all ───────────────────────────── #
def _run_all(video: Path) -> None:
    dst = CFG.paths.data_raw / "video.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(video, dst)
    logger.info(f"🎥 Видео скопировано → {dst}")

    extract(dst, CFG.paths.frames, CFG.extract.fps)
    augment_run()
    split_run()

    for sz in ("n", "s", "x"):
        _dispatch(train_main, ["--size", sz, "--epochs", str(CFG.train.epochs)])

    for best in CFG.paths.runs.rglob("best.pt"):
        _dispatch(eval_main, ["--weights", str(best)])

    report_main()


# ──────────────────────────── entrypoint ─────────────────────────────── #
def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "all":
        _run_all(args.video)

    elif args.cmd == "train":
        _dispatch(train_main, ["--size", args.size, "--epochs", str(args.epochs)])

    elif args.cmd == "eval":
        _dispatch(eval_main, ["--weights", str(args.weights)])

    elif args.cmd == "infer":
        # ——— 1. подставляем значения по умолчанию ——————————————— #
        video: Path = args.video or _default_video()
        weights: Path = args.weights or _default_weights()
        out: Path = args.out or Path(f"results/{video.stem}_boxes.mp4")

        logger.info("⚙️  Параметры инференса:"
                    f"\n   video   = {video}"
                    f"\n   weights = {weights}"
                    f"\n   out     = {out}"
                    f"\n   img     = {args.img}"
                    f"\n   conf    = {args.conf}"
                    f"\n   batch   = {args.batch}")

        out.parent.mkdir(parents=True, exist_ok=True)

        _dispatch(
            infer_video_main,
            [
                "--video", str(video),
                "--weights", str(weights),
                "--out", str(out),
                "--img", str(args.img),
                "--conf", str(args.conf),
                "--batch", str(args.batch),
            ],
        )

    elif args.cmd == "report":
        report_main()


if __name__ == "__main__":
    main()

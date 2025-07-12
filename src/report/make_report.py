# src/report/make_report.py
"""
Собирает Markdown-отчёт по всем экспериментам в runs/.
Запуск:  python -m src.report.make_report
"""

from __future__ import annotations

import json, re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)
plt.rcParams["figure.dpi"] = 150


# ───────────────────── helpers ──────────────────────
def col(fr: str, df: pd.DataFrame) -> str:
    """первый столбец, содержащий подстроку *fr* (без пробелов)"""
    for c in df.columns:
        if fr in c.replace(" ", ""):
            return c
    raise KeyError(fr)


def plot_history(run: Path, fig_dir: Path) -> list[str]:
    df = pd.read_csv(run / "results.csv")

    figs: list[str] = []

    # 1️⃣ loss & precision
    f1 = plt.figure()
    plt.plot(df[col("train/box_loss", df)], label="train box")
    plt.plot(df[col("metrics/precision", df)], label="val precision")
    plt.title("Loss & Precision")
    plt.xlabel("epoch"); plt.legend()
    fp1 = fig_dir / f"{run.name}_loss_prec.png"
    f1.savefig(fp1); plt.close(f1)
    figs.append(fp1)

    # 2️⃣ mAP50-95
    f2 = plt.figure()
    plt.plot(df[col("metrics/mAP50-95", df)], label="mAP50-95")
    plt.title("mAP50-95"); plt.xlabel("epoch"); plt.legend()
    fp2 = fig_dir / f"{run.name}_map.png"
    f2.savefig(fp2); plt.close(f2)
    figs.append(fp2)

    return figs


def read_metrics(run: Path) -> dict:
    """возвращает словарь базовых метрик для модели"""
    json_path = run / "results.json"
    if json_path.exists():
        with open(json_path) as f:
            j = json.load(f)
        # в results.json → список по эпохам; берём строку best (index -1)
        best = j[-1]
        return {
            "mAP50": round(best["metrics/mAP50(B)"], 4),
            "mAP50-95": round(best["metrics/mAP50-95(B)"], 4),
            "precision": round(best["metrics/precision(B)"], 4),
            "recall": round(best["metrics/recall(B)"], 4),
            "epochs": len(j),
        }

    # fallback: берём последнюю строку results.csv
    df = pd.read_csv(run / "results.csv")
    last = df.iloc[-1]
    return {
        "mAP50": round(last[col("metrics/mAP50(B)", df)], 4),
        "mAP50-95": round(last[col("metrics/mAP50-95(B)", df)], 4),
        "precision": round(last[col("metrics/precision(B)", df)], 4),
        "recall": round(last[col("metrics/recall(B)", df)], 4),
        "epochs": len(df),
    }


# ───────────────────── main ─────────────────────────
def main() -> None:
    report_md = CFG.paths.root / "report" / "report.md"
    fig_dir = CFG.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    with report_md.open("w") as rep:
        rep.write("# Итоговый отчёт\n\n")

        runs = sorted((CFG.paths.runs).glob("exp*"))
        if not runs:
            log.warning("Каталоги runs/exp* не найдены")
            return

        for run in runs:
            try:
                m = read_metrics(run)
            except Exception as e:
                log.warning(f"{run.name}: пропуск ({e})")
                continue

            size = re.search(r"exp[^_]*_(\w)", run.name)
            size = size.group(1) if size else "?"
            rep.write(f"## Модель **{size}**  ({run.name})\n\n")
            for k, v in m.items():
                rep.write(f"- **{k}**: {v}\n")
            rep.write("\n")

            imgs = plot_history(run, fig_dir)
            for img in imgs:
                rep.write(f"![{img.name}](figures/{img.name})\n\n")

    log.info(f"Report ready → {report_md}")


if __name__ == "__main__":
    main()

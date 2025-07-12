# src/utils/metrics.py

"""
metrics.py
Функции для приведения вывода Ultralytics к «плоскому» JSON + вспомогательные расчёты.
"""

from __future__ import annotations

import numpy as np


def _scalar(x):
    """
    Гарантирует float-скаляр из всего, что возвращает Ultralytics:
    • обычный float / int            → 그대로
    • ndarray / Tensor / list        → берём mean()
    • None                           → 0.0
    """
    if x is None:
        return 0.0
    # уже float / int
    if isinstance(x, (float, int)):
        return float(x)
    # у массивов и Tensor есть .mean()
    if hasattr(x, "mean"):
        return float(x.mean())
    # на всякий случай
    return float(np.asarray(x).mean())


def summarise_metrics(res):
    """
    От Ultralytics приходит DetMetrics со множеством полей.
    Выбираем ключевые метрики и округляем до 4 знаков.
    """
    box = res.box  # DetMetrics.box — то, где лежат mp, mr, map…
    grab = lambda *names: next(getattr(box, n, None) for n in names if hasattr(box, n))

    return {
        "mAP50":     round(_scalar(grab("map50", "ap50")), 4),
        "mAP50-95":  round(_scalar(grab("map",   "ap")),   4),
        "precision": round(_scalar(grab("precision", "mp")), 4),
        "recall":    round(_scalar(grab("recall",    "mr")), 4),
        "f1":        round(_scalar(grab("f1")), 4),
        # сколько изображений было у валидатора
        "dataset_size": getattr(res, "dataset", None).n if hasattr(res, "dataset") else None,
    }

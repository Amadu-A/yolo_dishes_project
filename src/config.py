# src/config.py

"""
config.py
Глобальная конфигурация проекта в виде dataclass.
Она импортируется из всех остальных модулей, чтобы
не дублировать параметры (пути, гиперпараметры и т. д.).
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Paths:
    """Собирает все пути проекта в одном месте."""
    root: Path = Path(__file__).resolve().parents[1]                          # корень репозитория
    data_raw: Path = root / "data" / "raw"                                    # сырые данные
    data_processed: Path = root / "data" / "processed"                        # после аугментации
    dataset: Path = root / "data" / "dataset"                                 # готовый датасет YOLO
    frames: Path = data_raw / "frames"                                        # кадры из видео
    labels: Path = data_raw / "labels"                                        # разметка кадров
    figures: Path = root / "report" / "figures"                               # графики
    runs: Path = root / "runs"                                                # лог-директория Ultralytics

@dataclass
class ExtractConfig:
    """Параметры извлечения кадров."""
    fps: int = 2                                                               # сколько кадров в секунду сохранять
    img_suffix: str = ".jpg"                                                   # расширение выходных файлов

@dataclass
class TrainConfig:
    """Гиперпараметры обучения YOLOv11."""
    model_size: str = "s"                                                      # s | n | x
    img_size: int = 640                                                        # сторона квадрата входного изображения
    epochs: int = 50                                                           # эпох обучения
    batch: int = 16                                                            # размер батча
    lr0: float = 0.01                                                          # начальная learning rate
    device: str = "0"                                                          # GPU id, "cpu" если без видеокарты

@dataclass
class ProjectConfig:
    """Корневой контейнер для всех групп параметров."""
    paths: Paths = field(default_factory=Paths)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

# Экземпляр, который удобно импортировать
CFG = ProjectConfig()

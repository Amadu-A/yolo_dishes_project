# src/utils/logger.py

"""
logger.py
Настраиваем единый логер для всего проекта.
Используем стандартный logging; формат единый, время в 24-часовом ISO-формате.
"""

import logging
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Возвращает настроенный логгер.

    :param name: название логгера (обычно __name__)
    :param log_file: путь к файлу (если None — лог только в консоль)
    """
    logger = logging.getLogger(name)  # создаём/берём логгер по имени
    if logger.handlers:  # предотвращаем дублирование хендлеров
        return logger

    logger.setLevel(logging.INFO)  # уровень INFO достаточно для пайплайна
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()  # вывод в stdout
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)  # убеждаемся, что папка логов есть
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

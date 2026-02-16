import logging
import os
from datetime import date
from pathlib import Path


class DailyFileHandler(logging.Handler):
    """
    Write logs to a date-stamped file (e.g., backend.log.2026-02-10).
    The handler switches files at local midnight.
    """

    def __init__(self, log_dir: Path, base_name: str, encoding: str = "utf-8") -> None:
        super().__init__()
        self._log_dir = log_dir
        self._base_name = base_name
        self._encoding = encoding
        self._current_date = date.today()
        self._handler = logging.FileHandler(
            self._path_for_date(self._current_date),
            encoding=self._encoding,
        )

    def _path_for_date(self, day: date) -> str:
        return str(self._log_dir / f"{self._base_name}.{day:%Y-%m-%d}")

    def _rollover_if_needed(self) -> None:
        today = date.today()
        if today == self._current_date:
            return
        self._handler.close()
        self._current_date = today
        self._handler = logging.FileHandler(
            self._path_for_date(self._current_date),
            encoding=self._encoding,
        )
        if self.formatter:
            self._handler.setFormatter(self.formatter)
        self._handler.setLevel(self.level)

    def setFormatter(self, fmt: logging.Formatter | None) -> None:  # type: ignore[override]
        super().setFormatter(fmt)
        self._handler.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._rollover_if_needed()
            self._handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._handler.close()
        super().close()


def init_logging() -> None:
    log_dir = Path(os.getenv("LOG_DIR", "api/logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = DailyFileHandler(
        log_dir=log_dir,
        base_name="backend.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

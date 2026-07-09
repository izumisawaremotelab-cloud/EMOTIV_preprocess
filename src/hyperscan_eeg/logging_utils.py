"""共通ロギング設定。"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """ルートロガーを設定する。全モジュールは `logging.getLogger(__name__)` を使う。

    Args:
        level: ロギングレベル。
        log_file: 指定した場合、標準出力に加えてこのファイルにも出力する。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

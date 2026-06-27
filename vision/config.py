"""Typed configuration, loaded from ``config.toml``.

Tier 1 only needs ``model`` and ``effort``. Later tiers add nested sections
(``[voice]``, ``[heartbeat]``, ``[rails]``, ``[cost]``); ``extra="ignore"`` means
a richer config file won't break an earlier tier, and a missing file falls back
to sensible defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "claude-opus-4-8"
    effort: Effort = "high"


def load_config(path: str | Path = "config.toml") -> Config:
    """Load config from TOML, or return defaults if the file is absent."""
    p = Path(path)
    if not p.exists():
        return Config()
    with p.open("rb") as f:
        data = tomllib.load(f)
    return Config(**data)

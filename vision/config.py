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

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class VoiceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ptt_key: str = "space"          # hold-to-talk key
    input_device: str = ""          # "" = system default microphone
    output_device: str = ""         # "" = system default speaker
    stt_language: str = "en-US"
    tts_voice_id: str = ""          # chosen at Tier 3; empty = adapter default


class QuietHours(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: str = "22:00"   # HH:MM local; non-urgent surfacing waits until `end`
    end: str = "07:00"


class CheckConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    interval_seconds: int = 300
    handler: str          # resolved to a registered check handler by name


class HeartbeatConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interval_seconds: int = 60          # how often the loop ticks
    confirm_timeout_seconds: int = 120  # heartbeat confirmations time out → deny
    quiet_hours: QuietHours = Field(default_factory=QuietHours)
    checks: list[CheckConfig] = Field(default_factory=list)


class RailsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Tools that always require confirmation, by name — the "never without asking"
    # list. (Tools may also self-flag via requires_confirmation.)
    consequential_tools: list[str] = Field(
        default_factory=lambda: [
            "send_email",
            "send_message",
            "send_social_post",
            "spend_money",
            "delete_reminder",
            "edit_file",
            "forget_fact",
        ]
    )
    kill_switch: bool = False  # initial state; persisted thereafter


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_per_mtok: float = 5.0    # claude-opus-4-8 pricing
    output_per_mtok: float = 25.0
    budget_warn_usd: float = 5.0


class ClaudeCodeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # acceptEdits = let it edit files (safe default); bypassPermissions = full
    # autonomy incl. running commands; default = ask (won't work unattended).
    permission_mode: str = "acceptEdits"
    max_turns: int = 30
    timeout_seconds: int = 600


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "claude-opus-4-8"
    effort: Effort = "high"
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    rails: RailsConfig = Field(default_factory=RailsConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    claude_code: ClaudeCodeConfig = Field(default_factory=ClaudeCodeConfig)


def load_config(path: str | Path = "config.toml") -> Config:
    """Load config from TOML, or return defaults if the file is absent."""
    p = Path(path)
    if not p.exists():
        return Config()
    with p.open("rb") as f:
        data = tomllib.load(f)
    return Config(**data)

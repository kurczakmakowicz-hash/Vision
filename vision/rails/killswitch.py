"""The kill switch — one obvious way to pause all proactive behavior at once.

Engaging it halts the heartbeat and holds background actions, while leaving the
chat fully usable. State is persisted, so a pause survives a restart (you stay in
control until you explicitly resume). You want this the first time Vision does
something unexpected — not after.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _default_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "killswitch.json"


class KillSwitch:
    def __init__(self, path: str | Path | None = None, default: bool = False) -> None:
        self.path = Path(path) if path else _default_path()
        self._engaged = default
        if self.path.exists():
            try:
                self._engaged = bool(json.loads(self.path.read_text()).get("engaged", default))
            except (json.JSONDecodeError, OSError):
                pass

    @property
    def engaged(self) -> bool:
        return self._engaged

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"engaged": self._engaged}))

    def engage(self) -> None:
        self._engaged = True
        self._save()

    def release(self) -> None:
        self._engaged = False
        self._save()

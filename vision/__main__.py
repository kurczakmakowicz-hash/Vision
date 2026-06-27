"""Entry point: `python -m vision` or the `vision` console script."""

from __future__ import annotations

import asyncio

from vision.app import main


def main_sync() -> None:
    """Synchronous wrapper so this works as a console_scripts entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()

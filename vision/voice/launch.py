"""Start a voice session from the text REPL, degrading gracefully.

Voice is opt-in (the `/voice` command). If the ``voice`` extra isn't installed,
or the STT/TTS keys are missing, or there's no audio device, this says so plainly
and returns to the text REPL — which always stays alive.
"""

from __future__ import annotations

import os

from vision.config import Config
from vision.core.conversation import Conversation
from vision.seams.provider.base import Provider
from vision.tools.registry import Registry

_VOICE_DEPS = ("sounddevice", "numpy", "deepgram", "elevenlabs", "pynput")


def _missing_deps() -> list[str]:
    import importlib.util

    return [m for m in _VOICE_DEPS if importlib.util.find_spec(m) is None]


async def start_voice(
    conversation: Conversation,
    provider: Provider,
    registry: Registry | None,
    config: Config,
) -> None:
    missing = _missing_deps()
    if missing:
        print(
            "[voice] Voice needs extra packages. Install them with:\n"
            "        pip install -e .[voice]\n"
            f"        (missing: {', '.join(missing)})\n"
        )
        return

    if not os.environ.get("DEEPGRAM_API_KEY") or not os.environ.get("ELEVENLABS_API_KEY"):
        print(
            "[voice] DEEPGRAM_API_KEY and ELEVENLABS_API_KEY must be set in .env to "
            "use voice. Staying in text mode.\n"
        )
        return

    # Imported here so the package never hard-depends on the voice extra.
    from vision.seams.stt.deepgram import DeepgramSTT
    from vision.seams.tts.elevenlabs import ElevenLabsTTS
    from vision.voice.playback import Playback
    from vision.voice.session import VoiceSession

    vc = config.voice
    session = VoiceSession(
        conversation,
        provider,
        registry,
        stt=DeepgramSTT(language=vc.stt_language),
        tts=ElevenLabsTTS(voice_id=vc.tts_voice_id or None),
        playback=Playback(vc.output_device or None),
        ptt_key=vc.ptt_key,
        input_device=vc.input_device or None,
        on_transcript=lambda t: print(f"\n[you said] {t}"),
        on_text=lambda d: print(d, end="", flush=True),
    )

    try:
        await session.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 — never let voice take down the REPL
        print(f"\n[voice] stopped: {exc}\n")
    finally:
        print("\n[back to text mode]\n")

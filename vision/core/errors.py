"""Errors the core understands, independent of any provider SDK.

Provider/STT/TTS adapters translate their vendor-specific exceptions into these
so the core and UI never import a vendor's error types.
"""

from __future__ import annotations


class VisionError(Exception):
    """Base class for everything Vision raises on purpose."""


class ProviderUnavailable(VisionError):
    """The model couldn't be reached (network, timeout, overload, rate limit).

    Transient — the UI shows a clear message and keeps the prompt alive.
    """


class ProviderAuthError(ProviderUnavailable):
    """The provider rejected our credentials (missing/invalid/!permitted key).

    A subclass of :class:`ProviderUnavailable` so the same catch handles it, but
    carries a credentials-specific message.
    """


# --- Reserved for later tiers (defined now so imports are stable) -------------


class ToolError(VisionError):
    """A tool failed; surfaced to the model as a plain-language tool result."""


class ConfirmationDenied(VisionError):
    """The user (or a timed-out heartbeat gate) declined a consequential action."""


class KillSwitchOn(VisionError):
    """Proactive behavior is halted by the kill switch."""

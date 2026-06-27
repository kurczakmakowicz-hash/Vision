"""Swappable seams: the model provider, speech-to-text, and text-to-speech.

Each subpackage has a ``base.py`` defining a ``Protocol`` and one file per
concrete backend. The core never imports a vendor SDK directly — it talks to
these interfaces, so a model/STT/TTS backend can change in one file.
"""

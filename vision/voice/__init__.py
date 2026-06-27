"""Voice: the ears and mouth, wrapped around the *same* brain and tools.

Push-to-talk first — the most reliable path. Voice changes only how turns arrive
(transcribed speech) and leave (spoken aloud); the turn itself still flows
through :func:`vision.core.agent.run_turn`. Hardware/vendor adapters import their
libraries lazily, so importing this package never requires the ``voice`` extra.
"""

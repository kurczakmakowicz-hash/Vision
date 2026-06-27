# Vision

A voice-first AI assistant harness — the core that turns a language model into
something you can talk to out loud, that can *act* on your behalf through tools,
remembers you between conversations, and can reach out first when something is
worth your attention. Built tier by tier; each layer is independently runnable
and tested.

See `AGENT.md` for the spec (identity, capabilities, and the "never without
asking" safety list).

## Install

```bash
pip install -e .            # the brain (Tiers 1–6, text)
pip install -e ".[voice]"   # add voice (Tier 3): Deepgram + ElevenLabs + audio
pip install -e ".[dev]"     # tests
```

## Configure

```bash
cp .env.example .env        # then add your keys (this file is git-ignored)
```

- `ANTHROPIC_API_KEY` — required (the brain).
- `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` — required for voice.

Tune behavior in `config.toml` (model, effort, voice, heartbeat checks, quiet
hours, the consequential-tools list, cost rates) — no code edits.

## Run

```bash
vision            # or: python -m vision
```

Commands in the prompt:

| Command | What it does |
|---|---|
| `/voice` | Start push-to-talk (needs the `voice` extra + keys); hold the key to talk |
| `/notices` | Show what the heartbeat surfaced |
| `/dismiss <id>` | Clear a surfaced notice |
| `/cost` | Show the running model-cost tally |
| `/kill` / `/resume` | Pause / resume all proactive behavior (chat stays up) |
| `/quit` | Leave |

Try the heartbeat: with Vision running, `echo "ping" > var/trigger` — the next
tick surfaces it.

## How it's built

One shared agent core, many ways in and out. A typed turn, a spoken turn, and a
turn the heartbeat starts all flow through the same `run_turn`.

| Tier | What | Where |
|---|---|---|
| 1 | The brain — streaming text loop | `vision/core/`, `vision/seams/provider/` |
| 2 | The hands — tool registry + manual tool-use loop | `vision/tools/` |
| 3 | The ears & mouth — push-to-talk voice | `vision/voice/`, `vision/seams/{stt,tts}/` |
| 4 | The memory — durable facts across restarts | `vision/memory/` |
| 5 | The heartbeat — proactive, quiet by default | `vision/heartbeat/` |
| 6 | The rails — gate, injection defense, audit, cost, kill switch | `vision/rails/` |

Add a capability = one self-contained file in `vision/tools/` (it self-registers).
Swap the model/STT/TTS backend = one file implementing a `Protocol` in
`vision/seams/`. The core changes for neither.

## Safety

- Anything that sends, spends, or deletes stops for an explicit yes (the gate) —
  per-action, covering text, voice, and the heartbeat.
- Content read from the outside world is treated as data; injected instructions
  are flagged to you, never obeyed.
- A plain audit log (`var/audit.log`), a cost tally, and a kill switch.

## Test

```bash
pytest -q
```

The deterministic core (loop, tools, chunker, memory, heartbeat scheduling, gate,
injection, cost, kill switch) is covered by tests that run with no API key or
audio device. Live voice and live model calls need keys and hardware.

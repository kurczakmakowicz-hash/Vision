# Vision — spec

Single source of truth for what we're building and why. Captured from the Tier 0
interview. Edit this when intent changes; the running system prompt in
`vision/core/conversation.py` should stay consistent with the "Identity" section.

## Identity
- **Name:** Vision
- **What it's for:** A personal, voice-first AI assistant that can *act* on the
  user's behalf — not a chatbot demo. It remembers the user between
  conversations and can reach out first when something is genuinely worth their
  attention.
- **Audience:** Just the user (single-user for now; per-user state kept in mind
  but not built).
- **Personality / tone:** Warm, plain-spoken, and brief. Consistent everywhere
  (system prompt, greetings, logs).

## Stack
- **Language/runtime:** Python 3.11, `asyncio`-based. Small and dependency-light
  — no heavy framework.
- **Model:** `claude-opus-4-8` via the official Anthropic SDK, behind a thin
  swappable seam (`vision/seams/provider/`). Adaptive thinking; effort
  configurable.
- **Voice (Tier 3):** Deepgram for speech-to-text, ElevenLabs for
  text-to-speech, each behind its own seam. Push-to-talk first.
- **Runs:** Laptop-first. The heartbeat (Tier 5) is kept cleanly separable so it
  can relocate to an always-on host later without a rewrite.

## First capabilities (first tools / first test cases)
1. Coding help (read files, answer questions about code)
2. Editing code (edit files — consequential, gated)
3. Social-media management (later; posting is consequential, gated)
4. Reminders & calendar (capture/recall reminders; calendar later)

## Never without asking (hard confirmation gate — Tier 6)
A tool that does any of these must stop and get an explicit "yes" before it runs,
stating plainly what it intends to do. Per-action; approval never generalizes.
- Send a message or email
- Spend money
- Delete or overwrite data

(Routine config edits are *not* gated by request — but Vision will not silently
rewrite its own safety config without telling the user.)

## Proactivity (Tier 5)
Yes — Vision may surface reminders and noticed conditions on its own. But it is
**quiet by default**: it earns the right to interrupt, it does not assume it.

## How the user talks to it
Text first (always kept alive), push-to-talk in Tier 3, wake word much later.

## The one rule
Get the brain working in plain text before adding any audio. Voice is a thin
layer on top of a working agent — one shared agent core, many ways in and out.
Every turn (typed, spoken, or heartbeat-initiated) flows through the same
`run_turn`.

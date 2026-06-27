"""The heartbeat: a background loop that lets Vision act without being spoken to.

Separate from the conversation loop. It wakes on an interval, runs scheduled
checks, and routes anything noteworthy to one place the user sees — quietly by
default. State (next-due times and held notices) lives in a JSON file, so the
loop survives restarts and can relocate to an always-on host without a rewrite.
"""

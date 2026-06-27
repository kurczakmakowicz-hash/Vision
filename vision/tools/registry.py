"""Tool registry: declare a tool with a Pydantic input model + the ``@tool``
decorator, and it self-registers. The registry renders Anthropic ``input_schema``
from the model and dispatches calls. The core loop only ever asks the registry
for schemas and looks tools up by name.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

# A handler takes the validated input model and returns plain text for the model
# (sync or async). Errors are caught by the loop, not raised to the user.
Handler = Callable[[BaseModel], Awaitable[str]] | Callable[[BaseModel], str]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler = field(repr=False)
    # Decide, per tool, whether it's safe to run on its own. Anything that sends,
    # spends, deletes, or overwrites must NOT — flag it here; Tier 6's gate stops
    # it until the user confirms.
    requires_confirmation: bool = False
    # True if the tool returns content from the outside world (a file, a web page,
    # an email). Tier 6 scans these results for injected instructions.
    external_content: bool = False

    def api_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def api_schemas(self) -> list[dict[str, Any]]:
        return [s.api_schema() for s in self._specs.values()]


# --- declarative registration -------------------------------------------------

_REGISTERED: list[ToolSpec] = []


def tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    requires_confirmation: bool = False,
    external_content: bool = False,
) -> Callable[[Handler], Handler]:
    """Register the decorated function as a tool. Describe it for a reader, not a
    compiler — the model picks tools by their descriptions."""

    def decorator(fn: Handler) -> Handler:
        _REGISTERED.append(
            ToolSpec(
                name=name,
                description=description,
                input_model=input_model,
                handler=fn,
                requires_confirmation=requires_confirmation,
                external_content=external_content,
            )
        )
        return fn

    return decorator


def discover_tools() -> Registry:
    """Import every tool module (running its ``@tool`` decorators) and build a
    Registry. Idempotent — modules import once, so re-calling is cheap."""
    import vision.tools as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_") or mod.name == "registry":
            continue
        importlib.import_module(f"vision.tools.{mod.name}")

    registry = Registry()
    for spec in _REGISTERED:
        registry.register(spec)
    return registry

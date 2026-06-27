"""The hands: tools the agent can choose to call.

One file per capability. A tool registers itself with the ``@tool`` decorator;
``discover_tools()`` imports every module here so those decorators run, then
returns a populated :class:`~vision.tools.registry.Registry`. Adding a capability
never touches the core loop — write a tool file and it self-registers.
"""

"""Tests that plugins passed to StrandsAgent(plugins=...) are forwarded to per-thread instances.

Strands stores plugins in ``_plugin_registry`` (a PluginRegistry) after Agent
init — the original Plugin list is not retained, and there is no public API to
recover it. ``_extract_agent_kwargs`` therefore cannot auto-forward plugins.

StrandsAgent accepts an explicit ``plugins`` kwarg and forwards it to every
per-thread StrandsAgentCore so Plugin-based extensions (e.g. ContextOffloader)
actually execute on the agents that serve requests.

Each test below is written to FAIL on pre-fix code (plugins dropped) and
PASS once plugins are forwarded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.models.model import Model
from strands.tools.registry import ToolRegistry

from ag_ui_strands.agent import StrandsAgent


def _mock_model():
    m = MagicMock(spec=Model)
    m.stateful = False
    return m


def _run_input(thread_id: str = "t1"):
    from ag_ui.core import RunAgentInput, UserMessage

    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", content="hello")],
        tools=[],
        context=[],
        forwarded_props={},
    )


class _CapturingCore:
    """Replacement for StrandsAgentCore that records constructor kwargs."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.tool_registry = ToolRegistry()

    async def stream_async(self, _msg, **_kwargs):
        if False:
            yield


async def _trigger_thread_creation(ag: StrandsAgent, thread_id: str):
    from ag_ui.core import RunErrorEvent

    events = []
    async for ev in ag.run(_run_input(thread_id)):
        events.append(ev)
        if thread_id in ag._agents_by_thread:
            break
    run_errors = [ev for ev in events if isinstance(ev, RunErrorEvent)]
    assert not run_errors, (
        f"ag.run() emitted RunErrorEvent(s): {run_errors!r}"
    )
    instance = ag._agents_by_thread.get(thread_id)
    assert instance is not None, (
        f"per-thread agent for {thread_id!r} was not created"
    )
    return instance


@pytest.mark.asyncio
async def test_plugins_forwarded_to_per_thread_agent():
    """Plugin instances passed to StrandsAgent(plugins=...) must be forwarded
    to every per-thread StrandsAgentCore constructor call."""
    plugin = MagicMock(name="my-plugin")
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test", plugins=[plugin])

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _trigger_thread_creation(ag, "t1")

    assert "plugins" in instance.init_kwargs, (
        "plugins kwarg not passed to per-thread StrandsAgentCore — "
        "any Plugin registered on the wrapper will never execute."
    )
    assert plugin in instance.init_kwargs["plugins"], (
        f"plugin missing from per-thread plugins list; "
        f"got {instance.init_kwargs.get('plugins')}"
    )


@pytest.mark.asyncio
async def test_each_thread_gets_plugin():
    """Every per-thread agent must receive the same plugin instances."""
    plugin = MagicMock(name="my-plugin")
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test", plugins=[plugin])

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance_a = await _trigger_thread_creation(ag, "thread-a")
        instance_b = await _trigger_thread_creation(ag, "thread-b")

    assert plugin in instance_a.init_kwargs.get("plugins", [])
    assert plugin in instance_b.init_kwargs.get("plugins", [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugins_value,label",
    [(None, "plugins kwarg omitted (default)"), ([], "explicit empty list")],
    ids=["default-none", "explicit-empty"],
)
async def test_no_plugins_kwarg_omitted_for_falsy_input(plugins_value, label):
    """When no plugins are supplied, the ``plugins`` kwarg must be OMITTED
    entirely from per-thread StrandsAgentCore construction — not forwarded as
    ``None`` or ``[]``, which future Strands versions might misinterpret."""
    template = Agent(model=_mock_model())
    kwargs = {} if plugins_value is None else {"plugins": plugins_value}
    ag = StrandsAgent(template, name="test", **kwargs)

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _trigger_thread_creation(ag, "t1")

    assert "plugins" not in instance.init_kwargs, (
        f"[{label}] expected 'plugins' kwarg to be OMITTED from "
        f"StrandsAgentCore(**kwargs), but got {instance.init_kwargs.get('plugins')!r}"
    )


@pytest.mark.asyncio
async def test_plugins_kwarg_forwarded_when_plugin_supplied():
    """Positive-case complement: when at least one plugin IS supplied, the
    ``plugins=[...]`` kwarg must reach the per-thread StrandsAgentCore."""
    plugin = MagicMock(name="my-plugin")
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test", plugins=[plugin])

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _trigger_thread_creation(ag, "t1")

    assert "plugins" in instance.init_kwargs, (
        "'plugins' kwarg missing from StrandsAgentCore(**kwargs) even though "
        "a Plugin was supplied to StrandsAgent(plugins=[...])"
    )
    forwarded = instance.init_kwargs["plugins"]
    assert isinstance(forwarded, list), (
        f"expected 'plugins' to be a list, got {type(forwarded).__name__}"
    )
    assert plugin in forwarded, (
        f"expected plugin {plugin!r} to be forwarded; got {forwarded!r}"
    )


@pytest.mark.asyncio
async def test_plugins_and_hooks_forwarded_together():
    """plugins and hooks must coexist — supplying both must forward both."""
    from strands.hooks import HookProvider

    plugin = MagicMock(name="my-plugin")

    class _MinimalHooks(HookProvider):
        def register_hooks(self, registry):
            pass

    hook_provider = _MinimalHooks()
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test", plugins=[plugin], hooks=[hook_provider])

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _trigger_thread_creation(ag, "t1")

    assert plugin in instance.init_kwargs.get("plugins", [])
    assert hook_provider in instance.init_kwargs.get("hooks", [])

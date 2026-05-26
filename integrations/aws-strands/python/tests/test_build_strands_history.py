"""Tests for _build_strands_history — Bedrock message-history construction.

Scenarios covered:

1. Consecutive assistant messages with tool calls are merged into one
   assistant message, and all matching tool results are combined into
   one user message (Bedrock requires all toolResults for a single
   assistant turn in one user message).

2. Mismatched toolUseId / toolCallId pairs are dropped; the conversation
   continues without the broken pair rather than crashing with a
   ValidationException.

3. Orphaned tool-result messages (no preceding assistant tool_use) are
   silently dropped.

4. Normal single-call turns pass through unchanged.

5. Text-only assistant messages adjacent to tool-call assistant messages
   are merged correctly.
"""

from __future__ import annotations

import pytest
from ag_ui.core import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from ag_ui_strands.agent import _build_strands_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(text: str, id: str = "u1") -> UserMessage:
    return UserMessage(id=id, content=text)


def _asst(text: str = "", tool_calls: list | None = None, id: str = "a1") -> AssistantMessage:
    return AssistantMessage(id=id, content=text, tool_calls=tool_calls or [])


def _tool(tool_call_id: str, content: str = "ok", id: str = "t1") -> ToolMessage:
    return ToolMessage(id=id, content=content, tool_call_id=tool_call_id)


def _tc(name: str, tc_id: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=tc_id, function=FunctionCall(name=name, arguments=args))


def _tool_use_ids(bedrock_messages: list) -> list[str]:
    """Return all toolUseIds from toolUse blocks in assistant messages."""
    ids = []
    for m in bedrock_messages:
        if m.get("role") != "assistant":
            continue
        for block in m.get("content", []):
            if "toolUse" in block:
                ids.append(block["toolUse"]["toolUseId"])
    return ids


def _tool_result_ids(bedrock_messages: list) -> list[str]:
    """Return all toolUseIds from toolResult blocks in user messages."""
    ids = []
    for m in bedrock_messages:
        if m.get("role") != "user":
            continue
        for block in m.get("content", []):
            if "toolResult" in block:
                ids.append(block["toolResult"]["toolUseId"])
    return ids


# ---------------------------------------------------------------------------
# 1. Consecutive assistant tool-call messages are merged
# ---------------------------------------------------------------------------

class TestConsecutiveAssistantMerging:
    """
    The AG-UI client stores each parallel tool call as a separate assistant
    message.  _build_strands_history must merge them into one assistant
    message and put all matching results in one user message.
    """

    def _messages(self):
        return [
            _user("hello"),
            _asst("I'll call two tools."),
            _asst("", [_tc("tool_a", "id-a")]),
            _asst("", [_tc("tool_b", "id-b")]),
            _tool("id-a", "result-a"),
            _tool("id-b", "result-b"),
            _user("follow-up"),
        ]

    def test_produces_single_assistant_message_with_both_tool_uses(self):
        out = _build_strands_history(self._messages())
        use_ids = _tool_use_ids(out)
        assert sorted(use_ids) == sorted(["id-a", "id-b"]), (
            f"Expected both toolUseIds merged; got {use_ids}"
        )

    def test_produces_single_user_message_with_both_tool_results(self):
        out = _build_strands_history(self._messages())
        # There should be exactly one user message that contains two toolResult blocks.
        tool_result_user_msgs = [
            m for m in out
            if m.get("role") == "user"
            and any("toolResult" in b for b in m.get("content", []))
        ]
        assert len(tool_result_user_msgs) == 1, (
            f"Expected one user message with tool results, got {len(tool_result_user_msgs)}"
        )
        result_ids = [
            b["toolResult"]["toolUseId"]
            for b in tool_result_user_msgs[0]["content"]
        ]
        assert sorted(result_ids) == sorted(["id-a", "id-b"])

    def test_assistant_text_preserved_in_merged_message(self):
        out = _build_strands_history(self._messages())
        asst_msgs = [m for m in out if m.get("role") == "assistant"]
        # The text "I'll call two tools." should appear in the merged message.
        texts = [
            b.get("text", "")
            for m in asst_msgs
            for b in m.get("content", [])
            if "text" in b
        ]
        assert any("I'll call two tools" in t for t in texts), (
            f"Text was lost in merged assistant message; texts={texts}"
        )

    def test_follow_up_user_message_present(self):
        out = _build_strands_history(self._messages())
        plain_user = [
            m for m in out
            if m.get("role") == "user"
            and all("toolResult" not in b for b in m.get("content", []))
        ]
        assert len(plain_user) == 2, (
            f"Expected 2 plain user messages (hello + follow-up), got {len(plain_user)}"
        )

    def test_bedrock_alternating_roles(self):
        """Every assistant message must be followed by a user message."""
        out = _build_strands_history(self._messages())
        for idx, m in enumerate(out[:-1]):
            nxt = out[idx + 1]
            if m.get("role") == "assistant" and any(
                "toolUse" in b for b in m.get("content", [])
            ):
                assert nxt.get("role") == "user", (
                    f"Assistant with toolUse at {idx} not followed by user; "
                    f"got role={nxt.get('role')!r}"
                )


# ---------------------------------------------------------------------------
# 2. Mismatched toolUseId / toolCallId pairs are dropped
# ---------------------------------------------------------------------------

class TestMismatchedToolCallPair:
    """
    When the stored tool_call_id in a tool message does not match the tc.id
    in the preceding assistant message, both sides should be dropped so Bedrock
    never sees an unmatched pair.
    """

    def _messages(self):
        return [
            _user("hello"),
            _asst("", [_tc("tool_x", "id-correct")]),
            _tool("id-wrong", "result"),  # ID mismatch
            _user("follow-up"),
        ]

    def test_mismatched_pair_dropped(self):
        out = _build_strands_history(self._messages())
        assert _tool_use_ids(out) == [], (
            "Mismatched toolUse block should be dropped"
        )
        assert _tool_result_ids(out) == [], (
            "Mismatched toolResult block should be dropped"
        )

    def test_conversation_continues_after_drop(self):
        out = _build_strands_history(self._messages())
        user_texts = [
            b.get("text", "")
            for m in out if m.get("role") == "user"
            for b in m.get("content", []) if "text" in b
        ]
        assert any("hello" in t for t in user_texts)
        assert any("follow-up" in t for t in user_texts)

    def test_partial_match_keeps_matched_drops_unmatched(self):
        """One matched pair and one mismatched pair — only the matched pair survives."""
        msgs = [
            _user("hi"),
            _asst("", [_tc("tool_a", "id-a"), _tc("tool_b", "id-b")]),
            _tool("id-a", "result-a"),   # matches
            _tool("id-x", "result-x"),   # does NOT match id-b
            _user("next"),
        ]
        out = _build_strands_history(msgs)
        use_ids = _tool_use_ids(out)
        result_ids = _tool_result_ids(out)
        assert use_ids == ["id-a"], f"Expected only id-a toolUse; got {use_ids}"
        assert result_ids == ["id-a"], f"Expected only id-a toolResult; got {result_ids}"


# ---------------------------------------------------------------------------
# 3. Orphaned tool-result messages are dropped
# ---------------------------------------------------------------------------

class TestOrphanedToolResult:
    def test_orphaned_tool_result_dropped(self):
        msgs = [
            _user("hello"),
            _tool("dangling-id", "result"),  # no preceding assistant tool_use
            _user("follow-up"),
        ]
        out = _build_strands_history(msgs)
        assert _tool_result_ids(out) == [], (
            "Orphaned tool result should be dropped"
        )
        user_texts = [
            b.get("text", "")
            for m in out if m.get("role") == "user"
            for b in m.get("content", []) if "text" in b
        ]
        assert any("hello" in t for t in user_texts)
        assert any("follow-up" in t for t in user_texts)


# ---------------------------------------------------------------------------
# 4. Normal single-call turns pass through unchanged
# ---------------------------------------------------------------------------

class TestNormalSingleCallTurn:
    def test_single_tool_call_round_trip(self):
        msgs = [
            _user("hello"),
            _asst("", [_tc("search", "id-1", '{"q": "foo"}')]),
            _tool("id-1", "search results"),
            _user("thanks"),
        ]
        out = _build_strands_history(msgs)

        assert _tool_use_ids(out) == ["id-1"]
        assert _tool_result_ids(out) == ["id-1"]

        # Check args survived
        asst = next(m for m in out if m.get("role") == "assistant")
        tu = next(b["toolUse"] for b in asst["content"] if "toolUse" in b)
        assert tu["name"] == "search"
        assert tu["input"] == {"q": "foo"}

    def test_tool_result_content_preserved(self):
        msgs = [
            _user("go"),
            _asst("", [_tc("fetch", "id-2")]),
            _tool("id-2", '{"data": 42}'),
        ]
        out = _build_strands_history(msgs)
        user_msg = next(m for m in out if m.get("role") == "user" and
                        any("toolResult" in b for b in m.get("content", [])))
        result_text = user_msg["content"][0]["toolResult"]["content"][0]["text"]
        assert "42" in result_text


# ---------------------------------------------------------------------------
# 5. Text-only assistant adjacent to tool-call assistant
# ---------------------------------------------------------------------------

class TestTextAndToolCallMerging:
    """
    When a text-only assistant message immediately precedes a tool-call
    assistant message, they should be merged into one Bedrock message.
    """

    def test_text_preserved_in_merged_message(self):
        msgs = [
            _user("inspect field"),
            _asst("I'll look that up for you.", id="a1"),
            _asst("", [_tc("read_data", "id-r")], id="a2"),
            _tool("id-r", "the data"),
        ]
        out = _build_strands_history(msgs)

        asst_msgs = [m for m in out if m.get("role") == "assistant"]
        # All assistant content should be in one message.
        assert len(asst_msgs) == 1, (
            f"Expected 1 merged assistant message, got {len(asst_msgs)}"
        )
        content_types = {
            list(b.keys())[0] for b in asst_msgs[0]["content"]
        }
        assert "text" in content_types, "text block missing from merged message"
        assert "toolUse" in content_types, "toolUse block missing from merged message"

    def test_exact_bedrock_structure(self):
        """Verify exact Bedrock content block structure."""
        msgs = [
            _user("go"),
            _asst("Calling tool.", id="a1"),
            _asst("", [_tc("my_tool", "tc-99", '{"x": 1}')], id="a2"),
            _tool("tc-99", "done"),
        ]
        out = _build_strands_history(msgs)

        asst = next(m for m in out if m.get("role") == "assistant")
        assert asst["content"][0] == {"text": "Calling tool."}
        assert asst["content"][1] == {
            "toolUse": {
                "toolUseId": "tc-99",
                "name": "my_tool",
                "input": {"x": 1},
            }
        }

        user_with_result = next(
            m for m in out
            if m.get("role") == "user"
            and any("toolResult" in b for b in m.get("content", []))
        )
        assert user_with_result["content"][0] == {
            "toolResult": {
                "toolUseId": "tc-99",
                "content": [{"text": "done"}],
                "status": "success",
            }
        }


# ---------------------------------------------------------------------------
# 6. Exact replay of the real-world failing case from the bug report
# ---------------------------------------------------------------------------

class TestRealWorldFailingCase:
    """
    The production error: agent called cpq_read_quote_summary and
    cpq_list_quote_templates in consecutive assistant messages.  The client
    stored them as separate assistant messages; _build_strands_history must
    merge them so Bedrock sees one assistant turn followed by one user turn.
    """

    def _messages(self):
        return [
            _user(
                "Please inspect the \"Opportunity\" field on the quote edit screen.",
                id="msg-0",
            ),
            _asst(
                "I'll inspect the Opportunity field for you.",
                id="msg-1",
            ),
            _asst(
                "",
                [_tc("cpq_read_quote_summary", "tooluse_Bd4n", '{"id": 628}')],
                id="msg-2",
            ),
            _asst(
                "",
                [_tc("cpq_list_quote_templates", "tooluse_w8Hh", "{}")],
                id="msg-3",
            ),
            _tool("tooluse_Bd4n", '{"formId": "gatekeeper_quote_template"}', id="msg-4"),
            _tool("tooluse_w8Hh", '{"templates": []}', id="msg-5"),
            _asst("Now let me read the layout.", id="msg-6"),
        ]

    def test_no_consecutive_assistant_messages(self):
        out = _build_strands_history(self._messages())
        for idx in range(len(out) - 1):
            if out[idx].get("role") == "assistant" and out[idx + 1].get("role") == "assistant":
                pytest.fail(
                    f"Two consecutive assistant messages at positions {idx} and {idx+1}: "
                    f"{out[idx]!r}, {out[idx+1]!r}"
                )

    def test_both_tool_uses_in_single_assistant_message(self):
        out = _build_strands_history(self._messages())
        use_ids = _tool_use_ids(out)
        assert "tooluse_Bd4n" in use_ids
        assert "tooluse_w8Hh" in use_ids
        # Both in ONE message
        asst_msgs = [m for m in out if m.get("role") == "assistant"]
        merged = [
            m for m in asst_msgs
            if sum(1 for b in m["content"] if "toolUse" in b) == 2
        ]
        assert len(merged) == 1, (
            f"Expected exactly one assistant message with 2 toolUse blocks; "
            f"assistant messages: {asst_msgs}"
        )

    def test_both_tool_results_in_single_user_message(self):
        out = _build_strands_history(self._messages())
        result_user_msgs = [
            m for m in out
            if m.get("role") == "user"
            and any("toolResult" in b for b in m.get("content", []))
        ]
        assert len(result_user_msgs) == 1
        result_ids = [
            b["toolResult"]["toolUseId"]
            for b in result_user_msgs[0]["content"]
        ]
        assert sorted(result_ids) == ["tooluse_Bd4n", "tooluse_w8Hh"]

    def test_no_validation_exception_structure(self):
        """
        Assert that every toolUse in an assistant message has a matching
        toolResult in the IMMEDIATELY following user message — which is
        exactly what Bedrock validates.
        """
        out = _build_strands_history(self._messages())
        for idx, m in enumerate(out[:-1]):
            if m.get("role") != "assistant":
                continue
            use_ids = {b["toolUse"]["toolUseId"] for b in m["content"] if "toolUse" in b}
            if not use_ids:
                continue
            nxt = out[idx + 1]
            assert nxt.get("role") == "user", (
                f"Assistant with toolUse at {idx} not followed by user message"
            )
            result_ids = {
                b["toolResult"]["toolUseId"]
                for b in nxt.get("content", [])
                if "toolResult" in b
            }
            assert use_ids == result_ids, (
                f"toolUseId mismatch: assistant has {use_ids}, "
                f"following user has {result_ids}"
            )

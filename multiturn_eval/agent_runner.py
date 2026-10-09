"""Replay the scripted user turns of each test case against a live Foundry agent.

Each test case runs in its own server-side conversation so the agent keeps its own
multi-turn state. Every output item (MCP tool calls, tool results, assistant text) is
converted to the Foundry evaluation message schema so tool-aware evaluators can score
the full trajectory.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from .dataset import TestCase

log = logging.getLogger(__name__)

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@dataclass
class AgentInfo:
    name: str
    version: str
    instructions: str | None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReplayResult:
    case: TestCase
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    error: str | None = None


def get_agent_info(project_client, agent_name: str, version: str | None = None) -> AgentInfo:
    """Resolve the agent version to evaluate. Pinning a version keeps runs comparable."""
    if version:
        details = project_client.agents.get_version(agent_name=agent_name, agent_version=version)
    else:
        details = project_client.agents.get(agent_name=agent_name).versions["latest"]
    definition = details.definition
    instructions = getattr(definition, "instructions", None)
    tools = [t.as_dict() if hasattr(t, "as_dict") else dict(t) for t in (getattr(definition, "tools", None) or [])]
    return AgentInfo(name=agent_name, version=str(details.version), instructions=instructions, tools=tools)


def replay_cases(
    openai_client,
    agent: AgentInfo,
    cases: list[TestCase],
    max_workers: int = 4,
    max_tool_output_chars: int = 4000,
    keep_conversations: bool = False,
) -> list[ReplayResult]:
    def run(case: TestCase) -> ReplayResult:
        return _replay_one(openai_client, agent, case, max_tool_output_chars, keep_conversations)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(run, cases))


def _replay_one(openai_client, agent, case, max_tool_output_chars, keep_conversation) -> ReplayResult:
    result = ReplayResult(case=case)
    agent_ref = {"agent_reference": {"name": agent.name, "version": agent.version, "type": "agent_reference"}}
    conversation = openai_client.conversations.create(metadata={"test_case_id": case.id[:64], "purpose": "evaluation"})
    tool_defs: dict[str, dict[str, Any]] = {}
    try:
        for turn, user_text in enumerate(case.user_turns, start=1):
            result.messages.append({"role": "user", "content": user_text})
            response = with_retry(
                lambda: openai_client.responses.create(conversation=conversation.id, input=user_text, extra_body=agent_ref)
            )
            result.response_ids.append(response.id)
            if response.status != "completed":
                raise RuntimeError(f"turn {turn}: response status '{response.status}' ({response.error})")
            result.messages.extend(convert_output(response.output, tool_defs, max_tool_output_chars))
            log.info("[%s] turn %d/%d done", case.id, turn, len(case.user_turns))
    except Exception as exc:  # noqa: BLE001 - one failed case must not abort the batch
        result.error = f"{type(exc).__name__}: {exc}"
        log.error("[%s] replay failed: %s", case.id, result.error)
    finally:
        if not keep_conversation:
            try:
                openai_client.conversations.delete(conversation_id=conversation.id)
            except Exception:  # noqa: BLE001
                log.warning("[%s] could not delete conversation %s", case.id, conversation.id)
    result.tool_definitions = list(tool_defs.values())
    return result


def convert_output(output, tool_defs: dict, max_chars: int) -> list[dict[str, Any]]:
    """Convert Responses API output items to evaluation messages; collects MCP tool definitions."""
    messages: list[dict[str, Any]] = []
    pending_commentary = False
    for item in output:
        kind = item.type
        if kind == "mcp_list_tools":
            for tool in item.tools:
                tool_defs[tool.name] = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                }
        elif kind in ("mcp_call", "function_call"):
            call_id = getattr(item, "call_id", None) or item.id
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": call_id, "name": item.name, "arguments": _parse_json(item.arguments)}
                    ],
                }
            )
            if kind == "mcp_call":
                tool_output = item.error if item.error else (item.output or "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": [{"type": "tool_result", "tool_result": _truncate(tool_output, max_chars)}],
                    }
                )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": [{"type": "tool_result", "tool_result": _truncate(item.output, max_chars)}],
                }
            )
        elif kind == "message" and item.role == "assistant":
            text = "".join(c.text for c in item.content if getattr(c, "type", "") == "output_text")
            if text.strip():
                # A "commentary" preamble immediately followed by the final answer (no tool call in
                # between) is a near-duplicate of that answer; keep only the final answer.
                if _is_text(messages[-1:]) and getattr(item, "phase", None) == "final_answer" and pending_commentary:
                    messages.pop()
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                pending_commentary = getattr(item, "phase", None) == "commentary"
                continue
        # `reasoning` items are encrypted/opaque and are intentionally not evaluated.
        if kind != "reasoning":
            pending_commentary = False
    return messages


def _is_text(last: list[dict[str, Any]]) -> bool:
    return bool(last) and last[0]["role"] == "assistant" and last[0]["content"][0].get("type") == "text"


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _truncate(value: Any, max_chars: int) -> Any:
    # Large retrieval payloads inflate judge cost and can exceed context; keep a bounded slice.
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"


def with_retry(fn, attempts: int = 5, base_delay: float = 2.0):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except RETRYABLE as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("transient error (%s); retrying in %.0fs", type(exc).__name__, delay)
            time.sleep(delay)

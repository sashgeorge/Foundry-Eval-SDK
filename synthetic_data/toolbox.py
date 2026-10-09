"""Minimal MCP client for the agent's Foundry toolbox.

Used to (1) discover the full tool catalog behind the toolbox's `tool_search`/`call_tool`
indirection and (2) pull real data excerpts so generated scenarios reference members,
claims, providers, and benefits that actually exist in the agent's knowledge sources.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests
from azure.core.credentials import TokenCredential

log = logging.getLogger(__name__)

TOKEN_SCOPE = "https://ai.azure.com/.default"
# Broad capability queries used to enumerate tools hidden behind `tool_search`.
DISCOVERY_QUERIES = [
    "customer member lookup",
    "enrollment policy deductible",
    "explanation of benefits claims",
    "provider directory doctors",
    "benefits coverage knowledge base",
    "pharmacy prescriptions",
]


@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolboxClient:
    def __init__(self, server_url: str, credential: TokenCredential, timeout: int = 90):
        self._url = server_url
        self._credential = credential
        self._timeout = timeout
        self._ids = itertools.count(1)

    def _rpc(self, method: str, params: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
        for attempt in range(1, attempts + 1):
            try:
                return self._rpc_once(method, params)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == attempts:
                    raise
                log.warning("toolbox %s transient error (%s); retrying", method, type(exc).__name__)
                time.sleep(2 * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _rpc_once(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        token = self._credential.get_token(TOKEN_SCOPE).token
        response = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params},
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.text
        if body.lstrip().startswith(("event:", "data:")):
            body = "\n".join(line[5:] for line in body.splitlines() if line.startswith("data:"))
        payload = json.loads(body)
        if "error" in payload:
            raise RuntimeError(f"MCP {method} error: {payload['error']}")
        return payload["result"]

    @staticmethod
    def _text(result: dict[str, Any]) -> str:
        return "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")

    def list_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(t["name"], t.get("description", ""), t.get("inputSchema") or {"type": "object", "properties": {}})
            for t in self._rpc("tools/list", {}).get("tools", [])
        ]

    def discover_tools(self) -> list[ToolInfo]:
        """Directly exposed tools plus every tool reachable through `tool_search`."""
        tools = {t.name: t for t in self.list_tools()}
        if "tool_search" in tools:
            for query in DISCOVERY_QUERIES:
                text = self._text(self._rpc("tools/call", {"name": "tool_search", "arguments": {"query": query, "limit": 10}}))
                try:
                    found = json.loads(text).get("tools", [])
                except json.JSONDecodeError:
                    continue  # "No tools matched ..."
                for t in found:
                    tools.setdefault(
                        t["name"],
                        ToolInfo(t["name"], t.get("description", ""), t.get("inputSchema") or {"type": "object", "properties": {}}),
                    )
        return list(tools.values())

    def call(self, tool_name: str, arguments: dict[str, Any], direct_names: set[str]) -> str:
        if tool_name in direct_names:
            return self._text(self._rpc("tools/call", {"name": tool_name, "arguments": arguments}))
        return self._text(
            self._rpc("tools/call", {"name": "call_tool", "arguments": {"name": tool_name, "arguments": arguments}})
        )

    def collect_grounding(self, tools: list[ToolInfo], max_chars_per_tool: int = 4000) -> dict[str, str]:
        """Query each data tool with its own description to sample representative records."""
        direct = {t.name for t in self.list_tools()}
        grounding: dict[str, str] = {}
        for tool in tools:
            if tool.name in ("tool_search", "call_tool"):
                continue
            if "query" not in (tool.parameters.get("properties") or {}):
                continue
            try:
                text = self.call(tool.name, {"query": tool.description or tool.name}, direct)
            except Exception as exc:  # noqa: BLE001 - grounding is best effort per tool
                log.warning("Grounding query failed for %s: %s", tool.name, exc)
                continue
            if text.strip():
                grounding[tool.name] = text[:max_chars_per_tool]
        return grounding

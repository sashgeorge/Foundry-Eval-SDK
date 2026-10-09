"""Generate a synthetic multi-turn evaluation file (messages + tool calls) from a live agent.

Pipeline
  1. Read the agent's instructions and MCP toolbox from its Foundry definition.
  2. Discover the toolbox's tool catalog and sample real records from each data tool.
  3. Ask an LLM to write grounded test scenarios (members, claims, providers that exist).
  4. Run each scenario against the live agent with an LLM-simulated caller, capturing every
     tool call and tool result.
  5. Keep only conversations where the agent's FIRST reply asks for the caller's name or
     policy number (and makes no tool call before it has one), then write JSONL.

Usage (from the project root):
  python -m synthetic_data.generate_multiturn_data
  python -m synthetic_data.generate_multiturn_data --count 12 --output synthetic_data/output/my-file.jsonl
  python -m synthetic_data.generate_multiturn_data --scenarios synthetic_data/output/scenarios.json   # reuse scenarios
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from multiturn_eval import agent_runner, dataset
from multiturn_eval.config import PROJECT_ROOT, load_settings, resolve_deployment
from synthetic_data.toolbox import ToolboxClient

log = logging.getLogger("synthetic_data")

OUTPUT_DIR = PROJECT_ROOT / "synthetic_data" / "output"
IDENTITY_REQUEST = re.compile(r"\b(name|policy)\b", re.IGNORECASE)
CATEGORIES = [
    "explanation of benefits (EOB) walkthrough and line-item questions",
    "billing dispute leading to an appeal / live-agent transfer",
    "enrollment, deductible and out-of-pocket status",
    "finding a high-quality in-network provider at a reasonable price",
    "plan benefits and coverage (e.g. vision, dental, dependents)",
    "caller who cannot be found in member records",
    "question the tools cannot answer (agent should decline and offer a live agent)",
    "upset or injured caller who needs empathy before information",
]

SCENARIO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenarios"],
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id", "category", "member_name", "policy_number", "identify_with", "opening_message",
                    "caller_facts", "persona", "desired_num_turns", "test_case_description",
                    "expected_outcome", "evaluation_criteria",
                ],
                "properties": {
                    "id": {"type": "string", "description": "snake_case unique id"},
                    "category": {"type": "string"},
                    "member_name": {"type": "string"},
                    "policy_number": {"type": "string"},
                    "identify_with": {"type": "string", "enum": ["name", "policy_number", "both"]},
                    "opening_message": {"type": "string"},
                    "caller_facts": {"type": "string"},
                    "persona": {"type": "string"},
                    "desired_num_turns": {"type": "integer"},
                    "test_case_description": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                    "evaluation_criteria": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

USER_TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["message", "done"],
    "properties": {"message": {"type": "string"}, "done": {"type": "boolean"}},
}

SCENARIO_PROMPT = """You design multi-turn evaluation scenarios for a customer-facing AI agent.

## Agent system prompt
{instructions}

## Agent tools (name: description)
{tools}

## Real records returned by the agent's tools (use ONLY these facts)
{grounding}

## Task
Write exactly {count} distinct scenarios, spreading them across these categories:
{categories}

Rules:
- Use members, policy numbers, claims, providers, and benefit facts that appear in the records above.
  For the "cannot be found" category, use a plausible name/policy number that does NOT appear in the records.
- Do not design scenarios whose main goal depends on a tool marked UNAVAILABLE.
- `opening_message` is the caller's first message. It must be a plan, benefit, claim, enrollment, or provider
  request that requires looking up the caller's record, so the agent must ask for their name or policy number.
  It must NOT contain the caller's name, policy number, or other identifiers. Put out-of-scope or unanswerable
  questions in a LATER turn (in `caller_facts`), never in the opening message.
- `caller_facts`: everything the caller knows and may share when asked (name, policy number, document amounts,
  dates, provider, zip code), plus their underlying goal and follow-up questions in order.
- `persona`: tone and behavior (e.g. anxious, terse, detail-oriented).
- `desired_num_turns`: number of caller messages, between 4 and 8.
- `test_case_description`: 1-3 sentences describing the caller's situation and goal (simulation seed style).
- `expected_outcome`: what an ideal agent does across the whole conversation, grounded in the records.
- `evaluation_criteria`: 4-6 objective, checkable statements. The FIRST criterion must be:
  "Requests the caller's name or policy number before looking up member, enrollment, or claim details".
  Include at least one criterion about using tools for factual answers (no guessing) where applicable.
"""

USER_SIM_PROMPT = """You are role-playing a CALLER phoning the TLC Healthcare benefits hotline. Never act as the agent.

Scenario: {test_case_description}
Persona: {persona}
What you know and want (share details only when the agent asks or when natural): {caller_facts}
Identify yourself using: {identify_with} (member: {member_name}, policy number: {policy_number})

Rules:
- Write the caller's NEXT message only: natural, concise, one or two sentences.
- When asked to identify yourself, give only the identifier(s) listed above ({identify_with}).
  Give the other identifier only if the agent asks for it.
- Answer exactly what the agent asked; then pursue your next goal or follow-up question.
- Don't repeat questions the agent already answered well.
- Set "done": true (with a short closing message) when your goals are met, the agent has transferred you,
  or the agent has clearly said it cannot help further.
"""


@dataclass
class Generated:
    scenario: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", help="Agent name (default: first name in AGENT-NAMES)")
    p.add_argument("--agent-version", help="Agent version (default: AGENT-VERSION or latest)")
    p.add_argument("--count", type=int, default=8, help="Number of scenarios/conversations to generate")
    p.add_argument("--scenarios", type=Path, help="Reuse a scenarios.json file instead of generating new scenarios")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "EvalData-Healthcare-Synthetic.jsonl")
    p.add_argument("--attempts", type=int, default=2, help="Attempts per scenario when validation fails")
    p.add_argument("--min-turns", type=int, default=3, help="Minimum caller messages before the caller may end the call")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tool-chars", type=int, default=4000, help="Truncate tool results to this many characters")
    return p.parse_args()


def structured(openai_client, model: str, instructions: str, prompt: str, schema: dict, name: str) -> dict[str, Any]:
    response = agent_runner.with_retry(
        lambda: openai_client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        )
    )
    return json.loads(response.output_text)


def generate_scenarios(openai_client, model, agent, tools, grounding, count) -> list[dict[str, Any]]:
    def describe(t) -> str:
        unavailable = t.name not in grounding and t.name not in ("tool_search", "call_tool")
        return f"- {t.name}: {t.description}" + (" [UNAVAILABLE: returned no data or errored]" if unavailable else "")

    prompt = SCENARIO_PROMPT.format(
        instructions=agent.instructions,
        tools="\n".join(describe(t) for t in tools),
        grounding="\n\n".join(f"### {name}\n{text}" for name, text in grounding.items()),
        count=count,
        categories="\n".join(f"- {c}" for c in CATEGORIES),
    )
    result = structured(openai_client, model, "Return only JSON that matches the schema.", prompt, SCENARIO_SCHEMA, "scenarios")
    scenarios = result["scenarios"][:count]
    seen: set[str] = set()
    for s in scenarios:
        base, n = re.sub(r"[^a-z0-9_]+", "_", s["id"].lower()).strip("_") or "scenario", 2
        s["id"] = base
        while s["id"] in seen:
            s["id"], n = f"{base}_{n}", n + 1
        seen.add(s["id"])
        s["desired_num_turns"] = max(2, min(int(s["desired_num_turns"]), 10))
    return scenarios


def transcript_for_user(messages: list[dict[str, Any]]) -> str:
    """The caller only hears what the agent says, never tool calls or tool results."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"CALLER: {m['content']}")
        elif m["role"] == "assistant":
            text = " ".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")
            if text:
                lines.append(f"AGENT: {text}")
    return "\n".join(lines)


def identifiers_in(text: str, scenario: dict[str, Any]) -> list[str]:
    found = []
    policy = re.sub(r"\D", "", scenario["policy_number"])
    if policy and policy in re.sub(r"\D", "", text):
        found.append("policy number")
    name_parts = [p for p in re.split(r"\s+", scenario["member_name"]) if len(p) > 2]
    if name_parts and all(re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE) for p in name_parts):
        found.append("member name")
    return found


def validate_first_turn(scenario, first_turn: list[dict[str, Any]]) -> list[str]:
    errors = []
    leaked = identifiers_in(scenario["opening_message"], scenario)
    if leaked:
        errors.append(f"opening message already contains {', '.join(leaked)}")
    if any(c.get("type") == "tool_call" for m in first_turn if m["role"] == "assistant" for c in m["content"]):
        errors.append("agent called a tool before the caller identified themselves")
    reply = " ".join(
        c.get("text", "") for m in first_turn if m["role"] == "assistant" for c in m["content"] if c.get("type") == "text"
    )
    if not IDENTITY_REQUEST.search(reply):
        errors.append(f"first agent reply does not ask for name or policy number: {reply[:160]!r}")
    return errors


def run_conversation(openai_client, sim_model, agent, scenario, max_tool_chars, min_turns) -> Generated:
    out = Generated(scenario=scenario)
    min_turns = min(min_turns, scenario["desired_num_turns"])
    agent_ref = {"agent_reference": {"name": agent.name, "version": agent.version, "type": "agent_reference"}}
    tool_defs: dict[str, dict[str, Any]] = {}
    sim_instructions = USER_SIM_PROMPT.format(**scenario)
    conversation = openai_client.conversations.create(metadata={"purpose": "synthetic-eval-data", "scenario": scenario["id"][:64]})
    try:
        user_text = scenario["opening_message"]
        for turn in range(1, scenario["desired_num_turns"] + 1):
            out.messages.append({"role": "user", "content": user_text})
            response = agent_runner.with_retry(
                lambda: openai_client.responses.create(conversation=conversation.id, input=user_text, extra_body=agent_ref)
            )
            if response.status != "completed":
                raise RuntimeError(f"agent turn {turn} status {response.status}: {response.error}")
            turn_messages = agent_runner.convert_output(response.output, tool_defs, max_tool_chars)
            if not any(m["role"] == "assistant" and any(c.get("type") == "text" for c in m["content"]) for m in turn_messages):
                raise RuntimeError(f"agent turn {turn} returned no text reply")
            out.messages.extend(turn_messages)

            if turn == 1:
                out.errors = validate_first_turn(scenario, turn_messages)
                if out.errors:
                    return out
            if turn == scenario["desired_num_turns"]:
                break
            nxt = structured(
                openai_client, sim_model, sim_instructions,
                f"Conversation so far:\n{transcript_for_user(out.messages)}\n\nWrite the caller's next message.",
                USER_TURN_SCHEMA, "caller_turn",
            )
            if nxt["done"] and turn < min_turns:
                nxt = structured(
                    openai_client, sim_model, sim_instructions,
                    f"Conversation so far:\n{transcript_for_user(out.messages)}\n\n"
                    "Do not end the call yet. Ask your next natural follow-up question about your situation "
                    "(for example a detail, cost, next step, or one of your follow-up questions). Set done=false.",
                    USER_TURN_SCHEMA, "caller_turn",
                )
            if (nxt["done"] and turn >= min_turns) or not nxt["message"].strip():
                break
            user_text = nxt["message"].strip()
        log.info("[%s] %d messages, %d tool calls", scenario["id"], len(out.messages), count_tool_calls(out.messages))
    except Exception as exc:  # noqa: BLE001 - isolate failures per scenario
        out.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            openai_client.conversations.delete(conversation_id=conversation.id)
        except Exception:  # noqa: BLE001
            log.warning("[%s] could not delete conversation %s", scenario["id"], conversation.id)
    out.tool_definitions = list(tool_defs.values())
    return out


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant" for c in m["content"] if c.get("type") == "tool_call")


def to_row(g: Generated, catalog: list[dict[str, Any]], agent, generated_at: str) -> dict[str, Any]:
    s = g.scenario
    # Agent-visible tools first, then toolbox tools reached through call_tool.
    tool_definitions = {t["name"]: t for t in catalog}
    tool_definitions.update({t["name"]: t for t in g.tool_definitions})
    return {
        "id": s["id"],
        "category": s["category"],
        "test_case_description": s["test_case_description"],
        "desired_num_turns": s["desired_num_turns"],
        "conversation": {"messages": g.messages},
        "tool_definitions": list(tool_definitions.values()),
        "expected_outcome": s["expected_outcome"],
        "evaluation_criteria": s["evaluation_criteria"],
        "metadata": {
            "source": "synthetic",
            "agent_name": agent.name,
            "agent_version": agent.version,
            "member_name": s["member_name"],
            "policy_number": s["policy_number"],
            "user_turns": sum(1 for m in g.messages if m["role"] == "user"),
            "tool_calls": count_tool_calls(g.messages),
            "generated_at": generated_at,
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "httpx", "httpx2", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()
    settings = load_settings()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=settings.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        model = settings.judge_deployment or resolve_deployment(project_client, settings.model_name)
        sim_model = settings.simulator_deployment or model
        agent = agent_runner.get_agent_info(
            project_client, args.agent or settings.agent_names[0], args.agent_version or settings.agent_version
        )
        log.info("Agent %s v%s; generator model %s", agent.name, agent.version, model)

        mcp_servers = [t for t in agent.tools if t.get("type") == "mcp" and t.get("server_url")]
        tools, grounding = [], {}
        for server in mcp_servers:
            client = ToolboxClient(server["server_url"], credential)
            server_tools = client.discover_tools()
            tools.extend(server_tools)
            grounding.update(client.collect_grounding(server_tools))
        log.info("Discovered %d tools; grounding from %d tools", len(tools), len(grounding))
        unavailable = [t.name for t in tools if t.name not in grounding and t.name not in ("tool_search", "call_tool")]
        if unavailable:
            log.warning("Tools returning no data (scenarios will avoid them): %s", ", ".join(unavailable))
        catalog = [t.as_definition() for t in tools]

        if args.scenarios:
            scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
        else:
            scenarios = generate_scenarios(openai_client, model, agent, tools, grounding, args.count)
            scenarios_path = args.output.with_name(args.output.stem + ".scenarios.json")
            scenarios_path.parent.mkdir(parents=True, exist_ok=True)
            scenarios_path.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("Wrote %d scenarios to %s", len(scenarios), scenarios_path)

        def produce(scenario: dict[str, Any]) -> Generated:
            result = Generated(scenario=scenario)
            for attempt in range(1, args.attempts + 1):
                result = run_conversation(openai_client, sim_model, agent, scenario, args.max_tool_chars, args.min_turns)
                if not result.errors:
                    return result
                log.warning("[%s] attempt %d rejected: %s", scenario["id"], attempt, "; ".join(result.errors))
            return result

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(produce, scenarios))

    accepted = [r for r in results if not r.errors]
    rejected = [r for r in results if r.errors]
    dataset.write_jsonl([to_row(r, catalog, agent, generated_at) for r in accepted], args.output)

    # Round-trip through the evaluation loader to guarantee the file is consumable.
    dataset.load_test_cases(args.output)
    print(f"\nWrote {len(accepted)} conversation(s) to {args.output}")
    for r in accepted:
        print(f"  OK   {r.scenario['id']:<45} turns={sum(m['role'] == 'user' for m in r.messages):<2} "
              f"tool_calls={count_tool_calls(r.messages)}")
    for r in rejected:
        print(f"  DROP {r.scenario['id']:<45} {'; '.join(r.errors)[:200]}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())

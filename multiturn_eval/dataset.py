"""Load, validate, and reshape evaluation data into the Foundry evaluation schemas.

Source rows (e.g. Docs/EvalData-Healthcare-sample1.jsonl) look like:
    {"id", "test_case_description", "desired_num_turns",
     "conversation": {"messages": [...]}, "expected_outcome", "evaluation_criteria": [...]}

Foundry expects:
  * conversation datasets  -> one row per conversation with a `messages` array
  * simulation seed data   -> `test_case_id`, `test_case_description`, `simulation_configuration`
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from azure.core.exceptions import ResourceExistsError

MAX_SIMULATION_DESCRIPTION_CHARS = 2500
VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}


@dataclass
class TestCase:
    id: str
    description: str
    desired_num_turns: int | None
    messages: list[dict[str, Any]]
    expected_outcome: str
    evaluation_criteria: list[str]
    tool_definitions: list[dict[str, Any]] | None = None
    @property
    def user_turns(self) -> list[str]:
        return [_text_of(m) for m in self.messages if m.get("role") == "user"]


def load_test_cases(path: Path) -> list[TestCase]:
    if not path.exists():
        raise SystemExit(f"Evaluation data file not found: {path}")

    cases: list[TestCase] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue

        messages = (row.get("conversation") or {}).get("messages") or row.get("messages") or []
        case_id = str(row.get("id") or row.get("test_case_id") or f"case-{line_no}")
        problems = _validate_messages(messages)
        if case_id in seen_ids:
            problems.append(f"duplicate id '{case_id}'")
        if problems:
            errors.extend(f"line {line_no} ({case_id}): {p}" for p in problems)
            continue
        seen_ids.add(case_id)

        criteria = row.get("evaluation_criteria") or []
        if isinstance(criteria, str):
            criteria = [criteria]
        cases.append(
            TestCase(
                id=case_id,
                description=row.get("test_case_description", ""),
                desired_num_turns=row.get("desired_num_turns"),
                messages=messages,
                expected_outcome=row.get("expected_outcome", ""),
                evaluation_criteria=[str(c) for c in criteria],
                tool_definitions=row.get("tool_definitions") or None,
            )
        )

    if errors:
        raise SystemExit("Invalid evaluation data:\n  " + "\n  ".join(errors))
    if not cases:
        raise SystemExit(f"No test cases found in {path}")
    return cases


def _validate_messages(messages: list[dict[str, Any]]) -> list[str]:
    if not messages:
        return ["no conversation messages"]
    problems = [
        f"message {i} has invalid role '{m.get('role')}'"
        for i, m in enumerate(messages)
        if m.get("role") not in VALID_ROLES
    ]
    if not any(m.get("role") == "user" for m in messages):
        problems.append("conversation has no user turns")
    return problems


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and "text" in c)
    return ""


def render_transcript(messages: list[dict[str, Any]], max_tool_chars: int = 600) -> str:
    """Human-readable transcript used by the custom scenario-criteria judge."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?").upper()
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue
        for part in content or []:
            kind = part.get("type")
            if kind == "tool_call":
                lines.append(f"{role} [tool call] {part.get('name')}({json.dumps(part.get('arguments'))})")
            elif kind == "tool_result":
                result = json.dumps(part.get("tool_result"))
                if len(result) > max_tool_chars:
                    result = result[:max_tool_chars] + "...[truncated]"
                lines.append(f"TOOL [result]: {result}")
            elif "text" in part:
                lines.append(f"{role}: {part['text']}")
    return "\n".join(lines)


def to_conversation_row(
    case: TestCase,
    messages: list[dict[str, Any]],
    tool_definitions: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build one conversation-level evaluation row (`messages` schema + scenario fields)."""
    full_messages = list(messages)
    if system_prompt and not any(m.get("role") in ("system", "developer") for m in full_messages):
        full_messages.insert(0, {"role": "system", "content": system_prompt})

    row: dict[str, Any] = {
        "test_case_id": case.id,
        "messages": full_messages,
        # Scenario fields are rendered as strings so any judge prompt can consume them.
        "conversation_transcript": render_transcript([m for m in full_messages if m["role"] != "system"]),
        "expected_outcome": case.expected_outcome,
        "evaluation_criteria": "\n".join(f"- {c}" for c in case.evaluation_criteria),
    }
    if tool_definitions:
        row["tool_definitions"] = tool_definitions
    return row


def to_simulation_seed(case: TestCase, include_reference_facts: bool = True) -> dict[str, Any]:
    """Build a simulation seed row.

    The simulated user only knows what the description tells it. Appending the user's
    side of the reference conversation gives it realistic identifiers (name, policy
    number, EOB amounts) so the agent's lookup tools can actually find records.
    """
    description = case.description or case.expected_outcome
    if include_reference_facts:
        facts = " | ".join(case.user_turns)
        suffix = f" Details you may share when the agent asks: {facts}"
        description = (description + suffix)[:MAX_SIMULATION_DESCRIPTION_CHARS]

    seed: dict[str, Any] = {"test_case_id": case.id, "test_case_description": description}
    if case.desired_num_turns:
        seed["simulation_configuration"] = {"desired_num_turns": int(case.desired_num_turns)}
    return seed


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def upload_dataset(project_client, name: str, path: Path) -> str:
    """Upload a JSONL file as a Foundry dataset versioned by its content hash.

    Content-hash versions make uploads idempotent and give every eval run an exact,
    reproducible pointer to the data it scored.
    """
    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    try:
        dataset = project_client.datasets.upload_file(name=name, version=version, file_path=str(path))
    except ResourceExistsError:
        dataset = project_client.datasets.get(name=name, version=version)
    if not dataset.id:
        raise RuntimeError(f"Dataset {name}:{version} has no id")
    return dataset.id

"""Testing criteria (evaluators) for multi-turn conversation evaluation.

Two complementary evaluations are used:

1. Conversation quality (run with `evaluation_level="conversation"`): built-in evaluators
   score the whole `messages` array, including tool calls.
2. Scenario criteria (run at the default turn level): a custom prompt evaluator grades each
   row against its own `expected_outcome` and `evaluation_criteria`. Built-ins can't do this
   because the criteria differ per row. It runs at turn level because conversation-level
   prompt evaluators only receive `{{messages}}`; at turn level every mapped field is
   injected, and each row's transcript still covers the entire conversation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from azure.ai.projects.models import EvaluatorCategory, EvaluatorDefinitionType
from azure.core.exceptions import ResourceNotFoundError

log = logging.getLogger(__name__)

SCENARIO_EVALUATOR_NAME = "scenario-criteria-adherence"
# Custom evaluator CRUD is a preview API and requires an explicit opt-in header.
PREVIEW_HEADERS = {"Foundry-Features": "Evaluations=V1Preview"}
SCENARIO_EVALUATION_LEVELS = ["turn"]

SCENARIO_PROMPT = """
You are an expert QA reviewer for a customer-facing healthcare benefits assistant.
Grade the ASSISTANT's behavior across the ENTIRE multi-turn conversation against the
scenario's expected outcome and each evaluation criterion.

Rules:
- Judge only what the assistant actually said or did (including tool calls) in the transcript.
- A criterion is MET only if the transcript clearly shows it; partial or implied counts as NOT MET.
- Order-sensitive criteria (e.g. "verify identity before accessing claim details") must be met in that order.
- Penalize factual or arithmetic errors, invented policy details, and unsafe advice.

### Expected outcome
{{expected_outcome}}

### Evaluation criteria
{{evaluation_criteria}}

### Conversation transcript
{{conversation_transcript}}

### Scoring
result = (number of criteria MET) / (total number of criteria), as a float between 0 and 1.
If the conversation contradicts the expected outcome in a material way, cap result at 0.5.

### Output Format (JSON):
{
    "result": <float from 0 to 1>,
    "reason": "<one line per criterion: MET/NOT MET - short evidence; then one line on the expected outcome>"
}
""".strip()

SCENARIO_FIELDS = ("conversation_transcript", "expected_outcome", "evaluation_criteria")


def ensure_scenario_evaluator(project_client) -> tuple[str, str]:
    """Create the custom evaluator, or reuse an existing version with an identical definition.

    A fingerprint of the definition is stored in the version metadata so edits to the prompt
    or schema automatically produce a new evaluator version. Returns (name, version) so
    testing criteria can pin the exact version (reproducible, comparable runs).
    """
    definition = {
        "type": EvaluatorDefinitionType.PROMPT,
        "prompt_text": SCENARIO_PROMPT,
        "init_parameters": {
            "type": "object",
            "properties": {"deployment_name": {"type": "string"}, "threshold": {"type": "number"}},
            "required": ["deployment_name", "threshold"],
        },
        "data_schema": {
            "type": "object",
            "properties": {field: {"type": "string"} for field in SCENARIO_FIELDS},
            "required": list(SCENARIO_FIELDS),
        },
        "metrics": {
            "scenario_criteria": {
                "type": "continuous",
                "desirable_direction": "increase",
                "min_value": 0,
                "max_value": 1,
            }
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps({"definition": definition, "levels": SCENARIO_EVALUATION_LEVELS}, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]

    try:
        existing = list(
            project_client.evaluators.list_versions(name=SCENARIO_EVALUATOR_NAME, type="custom", headers=PREVIEW_HEADERS)
        )
    except ResourceNotFoundError:
        existing = []
    for version in existing:
        if (version.metadata or {}).get("fingerprint") == fingerprint:
            log.info("Reusing evaluator %s v%s", SCENARIO_EVALUATOR_NAME, version.version)
            return SCENARIO_EVALUATOR_NAME, str(version.version)

    created = project_client.evaluators.create_version(  # type: ignore[call-overload]
        name=SCENARIO_EVALUATOR_NAME,
        evaluator_version={
            "name": SCENARIO_EVALUATOR_NAME,
            "categories": [EvaluatorCategory.QUALITY],
            "display_name": "Scenario Criteria Adherence",
            "description": "Fraction of per-conversation evaluation criteria met across the whole conversation.",
            "supported_evaluation_levels": SCENARIO_EVALUATION_LEVELS,
            "metadata": {"fingerprint": fingerprint},
            "definition": definition,
        },
        headers=PREVIEW_HEADERS,
    )
    log.info("Created evaluator %s v%s", SCENARIO_EVALUATOR_NAME, created.version)
    return SCENARIO_EVALUATOR_NAME, str(created.version)


def _builtin(name: str, deployment: str, mapping: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "azure_ai_evaluator",
        "name": name,
        "evaluator_name": f"builtin.{name}",
        "initialization_parameters": {"deployment_name": deployment},
        "data_mapping": mapping,
    }


def conversation_quality_criteria(judge_deployment: str, *, has_tools: bool) -> list[dict[str, Any]]:
    """Built-in evaluators that support conversation-level scoring."""
    messages = {"messages": "{{item.messages}}"}
    with_tools = {**messages, "tool_definitions": "{{item.tool_definitions}}"} if has_tools else messages

    criteria = [
        # Composite: fluency, coherence, intent resolution, task adherence, groundedness, task completion.
        _builtin("output_quality", judge_deployment, with_tools),
        _builtin("customer_satisfaction", judge_deployment, messages),
        _builtin("task_completion", judge_deployment, with_tools),
    ]
    if has_tools:
        # Composite: tool call accuracy/success, input accuracy, output utilization, selection.
        criteria.append(_builtin("tool_use_quality", judge_deployment, with_tools))
    return criteria


def conversation_quality_schema(has_tools: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {"test_case_id": {"type": "string"}, "messages": {"type": "array"}}
    if has_tools:
        properties["tool_definitions"] = {"type": "array"}
    return {"type": "object", "properties": properties, "required": ["messages"]}


def scenario_criteria(judge_deployment: str, evaluator: tuple[str, str], threshold: float) -> list[dict[str, Any]]:
    name, version = evaluator
    return [
        {
            "type": "azure_ai_evaluator",
            "name": "scenario_criteria",
            "evaluator_name": name,
            "evaluator_version": version,
            "initialization_parameters": {"deployment_name": judge_deployment, "threshold": threshold},
            "data_mapping": {field: f"{{{{item.{field}}}}}" for field in SCENARIO_FIELDS},
        }
    ]


def scenario_schema() -> dict[str, Any]:
    properties = {"test_case_id": {"type": "string"}, **{field: {"type": "string"} for field in SCENARIO_FIELDS}}
    return {"type": "object", "properties": properties, "required": list(SCENARIO_FIELDS)}

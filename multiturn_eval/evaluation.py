"""Create evaluations and runs, wait for completion, and summarize results."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai.types.eval_create_params import DataSourceConfigCustom
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}


def get_or_create_evaluation(openai_client, name: str, item_schema: dict, testing_criteria: list[dict], metadata: dict):
    """Reuse an evaluation whose schema + criteria are unchanged so runs accumulate in one place.

    All runs under one evaluation share the same schema and metrics, which is what makes them
    comparable over time (agent versions, prompt changes) in the Foundry portal.
    """
    fingerprint = hashlib.sha256(
        json.dumps({"schema": item_schema, "criteria": testing_criteria}, sort_keys=True).encode()
    ).hexdigest()[:12]
    for existing in openai_client.evals.list(limit=100, order="desc"):
        if existing.name == name and (existing.metadata or {}).get("fingerprint") == fingerprint:
            log.info("Reusing evaluation %s (%s)", name, existing.id)
            return existing
    return openai_client.evals.create(
        name=name,
        data_source_config=DataSourceConfigCustom(type="custom", item_schema=item_schema, include_sample_schema=False),
        testing_criteria=testing_criteria,  # type: ignore[arg-type]
        metadata={**metadata, "fingerprint": fingerprint},
    )


def start_dataset_run(
    openai_client, eval_id: str, run_name: str, dataset_id: str, metadata: dict, evaluation_level: str | None
):
    return openai_client.evals.runs.create(
        eval_id=eval_id,
        name=run_name,
        metadata=metadata,
        data_source=CreateEvalJSONLRunDataSourceParam(type="jsonl", source=SourceFileID(type="file_id", id=dataset_id)),
        # "conversation" scores each conversation as a whole instead of only its final turn.
        extra_body={"evaluation_level": evaluation_level} if evaluation_level else None,
    )


def start_simulation_run(
    openai_client,
    eval_id: str,
    run_name: str,
    seeds_dataset_id: str,
    agent_name: str,
    agent_version: str,
    simulator_deployment: str,
    metadata: dict,
    max_num_turns: int = 12,
    conversation_repetitions: int = 1,
    sampling_params: dict[str, Any] | None = None,
):
    model_configuration: dict[str, Any] = {"model": simulator_deployment}
    if sampling_params:
        # Reasoning models (gpt-5.x, o-series) reject temperature/top_p; only send what's configured.
        model_configuration["sampling_params"] = sampling_params
    return openai_client.evals.runs.create(
        eval_id=eval_id,
        name=run_name,
        metadata=metadata,
        data_source={  # type: ignore[arg-type]
            "type": "azure_ai_user_conversation_simulation",
            "source": {"type": "file_id", "id": seeds_dataset_id},
            "target": {"type": "azure_ai_agent", "name": agent_name, "version": agent_version},
            "model_configuration": model_configuration,
            "default_simulation_configuration": {
                "max_num_turns": max_num_turns,
                "conversation_repetitions": conversation_repetitions,
                "enable_conversation_dataset_generation": True,
                "output_conversation_dataset_name": f"{agent_name}-simulated-conversations",
            },
        },
        extra_body={"evaluation_level": "conversation"},
    )


def wait_for_run(openai_client, eval_id: str, run_id: str, poll_seconds: int = 10, timeout_seconds: int = 3600):
    deadline = time.monotonic() + timeout_seconds
    while True:
        run = openai_client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
        if run.status in TERMINAL_STATUSES:
            return run
        if time.monotonic() > deadline:
            raise TimeoutError(f"Run {run_id} still '{run.status}' after {timeout_seconds}s")
        log.info("run %s status: %s", run_id, run.status)
        time.sleep(poll_seconds)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def collect_results(openai_client, eval_id: str, run) -> dict[str, Any]:
    items = list(openai_client.evals.runs.output_items.list(run_id=run.id, eval_id=eval_id))

    per_criterion: dict[str, dict[str, Any]] = defaultdict(lambda: {"passed": 0, "failed": 0, "other": 0, "scores": []})
    rows: list[dict[str, Any]] = []
    for item in items:
        datasource_item = _get(item, "datasource_item") or {}
        row = {
            "test_case_id": _get(datasource_item, "test_case_id") or _get(item, "id"),
            "status": _get(item, "status"),
            "results": [],
        }
        for r in _get(item, "results") or []:
            name = _get(r, "name")
            metric = _get(r, "metric") or name
            # Composite evaluators (output_quality, tool_use_quality) emit one primary result plus
            # one result per component metric under the same name; report each separately.
            key = name if metric == name else f"{name}.{metric}"
            passed = _get(r, "passed")
            score = _get(r, "score")
            status = _get(r, "status")
            label = _get(r, "label")
            stats = per_criterion[key]
            if status not in (None, "completed") or label in ("not_applicable", "skipped"):
                stats["other"] += 1  # skipped / not applicable / errored
            elif passed is True:
                stats["passed"] += 1
            elif passed is False:
                stats["failed"] += 1
            else:
                stats["other"] += 1
            if isinstance(score, (int, float)):
                stats["scores"].append(score)
            row["results"].append(
                {"name": key, "passed": passed, "score": score, "label": label, "status": status, "reason": _get(r, "reason")}
            )
        rows.append(row)

    summary = {}
    for name, stats in per_criterion.items():
        graded = stats["passed"] + stats["failed"]
        summary[name] = {
            "passed": stats["passed"],
            "failed": stats["failed"],
            "not_graded": stats["other"],
            "pass_rate": round(stats["passed"] / graded, 3) if graded else None,
            "mean_score": round(sum(stats["scores"]) / len(stats["scores"]), 3) if stats["scores"] else None,
        }

    counts = _get(run, "result_counts")
    return {
        "runs": [
            {
                "eval_id": eval_id,
                "run_id": run.id,
                "name": _get(run, "name"),
                "status": run.status,
                "report_url": _get(run, "report_url"),
                "result_counts": counts.model_dump() if hasattr(counts, "model_dump") else counts,
                "raw_output_items": [i.model_dump() if hasattr(i, "model_dump") else i for i in items],
            }
        ],
        "per_criterion": summary,
        "items": rows,
    }


def merge_results(parts: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Combine results from several runs (e.g. conversation quality + scenario criteria)."""
    merged: dict[str, Any] = {"metadata": metadata, "runs": [], "per_criterion": {}, "items": []}
    by_case: dict[str, dict[str, Any]] = {}
    for part in parts:
        merged["runs"].extend(part["runs"])
        merged["per_criterion"].update(part["per_criterion"])
        for row in part["items"]:
            target = by_case.setdefault(row["test_case_id"], {"test_case_id": row["test_case_id"], "results": []})
            target["results"].extend(row["results"])
    merged["items"] = list(by_case.values())
    return merged


def save_results(results: dict[str, Any], results_dir: Path, label: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{label}-{stamp}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def print_summary(results: dict[str, Any]) -> None:
    for run in results["runs"]:
        print(f"\nRun {run['name']} ({run['run_id']}) -> {run['status']}  counts={run['result_counts']}")
    width = max([len(n) for n in results["per_criterion"]] + [9]) + 2
    print(f"\n{'criterion':<{width}}{'pass':>6}{'fail':>6}{'n/a':>6}{'pass_rate':>11}{'mean':>8}")
    for name, s in sorted(results["per_criterion"].items()):
        rate = "-" if s["pass_rate"] is None else f"{s['pass_rate']:.0%}"
        mean = "-" if s["mean_score"] is None else f"{s['mean_score']:.2f}"
        print(f"{name:<{width}}{s['passed']:>6}{s['failed']:>6}{s['not_graded']:>6}{rate:>11}{mean:>8}")
    print()
    for row in results["items"]:
        for r in row["results"]:
            reason = (r["reason"] or "").replace("\n", " ")[:400]
            if r["status"] not in (None, "completed") or r["label"] in ("not_applicable", "skipped"):
                print(f"  SKIP [{row['test_case_id']}] {r['name']}: {reason}")
            elif r["passed"] is False:
                print(f"  FAIL [{row['test_case_id']}] {r['name']}: {reason}")
    for run in results["runs"]:
        if run.get("report_url"):
            print(f"\nPortal report ({run['name']}): {run['report_url']}")


def gate(results: dict[str, Any], min_pass_rate: float) -> list[str]:
    """Return gate violations for primary criteria (use as a CI/CD quality gate).

    Component metrics of composite evaluators (`name.metric`) are diagnostic only; the
    composite's primary result already reflects them.
    """
    problems = []
    for name, s in results["per_criterion"].items():
        if "." in name:
            continue
        if s["pass_rate"] is None:
            problems.append(f"{name} was not graded (skipped or errored) - check evaluator configuration")
        elif s["pass_rate"] < min_pass_rate:
            problems.append(f"{name} pass_rate={s['pass_rate']:.0%} < {min_pass_rate:.0%}")
    return problems

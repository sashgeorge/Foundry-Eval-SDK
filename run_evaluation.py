"""Evaluate a Foundry agent on multi-turn conversations.

Modes:
  replay    Re-run each conversation's user turns against the live agent, then score
            the agent's actual responses and tool calls. (default)
  recorded  Score the conversations stored in the eval data file as-is.
  simulate  Let Foundry play the user from `test_case_description` and score the
            generated conversations.

Examples:
  python run_evaluation.py
  python run_evaluation.py --mode recorded
  python run_evaluation.py --mode simulate --repetitions 2
  python run_evaluation.py --mode replay --agent-version 12 --fail-under 0.9
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from multiturn_eval import agent_runner, dataset, evaluation, evaluators
from multiturn_eval.config import Settings, load_settings, resolve_deployment

log = logging.getLogger("multiturn_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "recorded", "simulate"], default="replay")
    p.add_argument("--agent", help="Agent name (default: every agent in AGENT-NAMES)")
    p.add_argument("--agent-version", help="Agent version to evaluate (default: AGENT-VERSION or latest)")
    p.add_argument("--repetitions", type=int, default=1, help="simulate: conversations per seed scenario")
    p.add_argument("--max-turns", type=int, default=12, help="simulate: hard cap on turns per conversation")
    p.add_argument(
        "--simulator-temperature", type=float,
        help="simulate: sampling temperature for the simulated user (omit for reasoning models such as gpt-5.x)",
    )
    p.add_argument("--workers", type=int, default=4, help="replay: conversations replayed in parallel")
    p.add_argument("--keep-conversations", action="store_true", help="replay: keep server-side conversations")
    p.add_argument("--no-system-prompt", action="store_true", help="Do not prepend agent instructions to messages")
    p.add_argument("--fail-under", type=float, help="Exit 1 if any criterion pass rate is below this (0-1)")
    p.add_argument("--timeout", type=int, default=3600, help="Seconds to wait for each eval run")
    return p.parse_args()


def build_rows(args, openai_client, agent, cases, failures: list[str]) -> list[dict[str, Any]]:
    system_prompt = None if args.no_system_prompt else agent.instructions
    if args.mode == "recorded":
        rows = [dataset.to_conversation_row(c, c.messages, c.tool_definitions, system_prompt) for c in cases]
    else:
        replays = agent_runner.replay_cases(
            openai_client, agent, cases, max_workers=args.workers, keep_conversations=args.keep_conversations
        )
        failures.extend(f"{agent.name}: replay failed for {r.case.id}: {r.error}" for r in replays if r.error)
        rows = [
            dataset.to_conversation_row(r.case, r.messages, r.tool_definitions, system_prompt)
            for r in replays
            if not r.error
        ]
    if any(r.get("tool_definitions") for r in rows):
        # Every row must carry the field once evaluators map it.
        for r in rows:
            r.setdefault("tool_definitions", [])
    return rows


def evaluate_dataset(args, settings: Settings, project_client, openai_client, agent, judge, cases, stamp, failures):
    """Run conversation-quality and scenario-criteria evaluations over one uploaded dataset."""
    rows = build_rows(args, openai_client, agent, cases, failures)
    if not rows:
        failures.append(f"{agent.name}: no conversations to evaluate")
        return None

    label = f"{settings.eval_data_file.stem}-{agent.name}-{args.mode}"
    data_path = dataset.write_jsonl(rows, settings.datasets_dir / f"{label}.jsonl")
    data_id = dataset.upload_dataset(project_client, label, data_path)
    has_tools = any(r.get("tool_definitions") for r in rows)
    metadata = _metadata(agent, args.mode, settings)
    eval_metadata = {"agent_name": agent.name, "mode": args.mode}

    quality_eval = evaluation.get_or_create_evaluation(
        openai_client,
        f"{agent.name}-{args.mode}-conversation-quality",
        evaluators.conversation_quality_schema(has_tools),
        evaluators.conversation_quality_criteria(judge, has_tools=has_tools),
        eval_metadata,
    )
    scenario_eval = evaluation.get_or_create_evaluation(
        openai_client,
        f"{agent.name}-{args.mode}-scenario-criteria",
        evaluators.scenario_schema(),
        evaluators.scenario_criteria(
            judge, evaluators.ensure_scenario_evaluator(project_client), settings.criteria_threshold
        ),
        eval_metadata,
    )
    run_prefix = f"{agent.name}-v{agent.version}-{args.mode}-{stamp}"
    started = [
        (quality_eval, evaluation.start_dataset_run(
            openai_client, quality_eval.id, f"{run_prefix}-quality", data_id, metadata, "conversation")),
        (scenario_eval, evaluation.start_dataset_run(
            openai_client, scenario_eval.id, f"{run_prefix}-criteria", data_id, metadata, None)),
    ]
    return _finish(openai_client, agent, started, metadata, args.timeout, failures)


def evaluate_simulation(args, settings: Settings, project_client, openai_client, agent, judge, cases, stamp, failures):
    label = f"{settings.eval_data_file.stem}-simulation-seeds"
    seeds_path = dataset.write_jsonl([dataset.to_simulation_seed(c) for c in cases], settings.datasets_dir / f"{label}.jsonl")
    seeds_id = dataset.upload_dataset(project_client, label, seeds_path)
    metadata = _metadata(agent, args.mode, settings)

    # Simulated conversations carry no per-row expected outcome, so only built-ins apply.
    quality_eval = evaluation.get_or_create_evaluation(
        openai_client,
        f"{agent.name}-simulate-conversation-quality",
        evaluators.conversation_quality_schema(has_tools=False),
        evaluators.conversation_quality_criteria(judge, has_tools=False),
        {"agent_name": agent.name, "mode": args.mode},
    )
    run = evaluation.start_simulation_run(
        openai_client, quality_eval.id, f"{agent.name}-v{agent.version}-simulate-{stamp}", seeds_id,
        agent.name, agent.version, settings.simulator_deployment or judge, metadata,
        max_num_turns=args.max_turns, conversation_repetitions=args.repetitions,
        sampling_params={"temperature": args.simulator_temperature} if args.simulator_temperature is not None else None,
    )
    return _finish(openai_client, agent, [(quality_eval, run)], metadata, args.timeout, failures)


def _finish(openai_client, agent, started, metadata, timeout, failures):
    parts = []
    for eval_obj, run in started:
        log.info("Waiting for eval %s run %s ...", eval_obj.name, run.id)
        run = evaluation.wait_for_run(openai_client, eval_obj.id, run.id, timeout_seconds=timeout)
        if run.status != "completed":
            failures.append(f"{agent.name}: run {run.id} ended '{run.status}': {getattr(run, 'error', None)}")
            continue
        part = evaluation.collect_results(openai_client, eval_obj.id, run)
        errored = (part["runs"][0]["result_counts"] or {}).get("errored") or 0
        if errored:
            failures.append(f"{agent.name}: {errored} item(s) errored in {eval_obj.name}")
        parts.append(part)
    return evaluation.merge_results(parts, metadata) if parts else None


def _metadata(agent, mode: str, settings: Settings) -> dict[str, str]:
    return {"agent_name": agent.name, "agent_version": agent.version, "mode": mode, "data": settings.eval_data_file.name}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("azure", "httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = parse_args()
    settings = load_settings()
    gate_threshold = args.fail_under if args.fail_under is not None else settings.pass_rate_gate
    cases = dataset.load_test_cases(settings.eval_data_file)
    log.info("Loaded %d test case(s) from %s", len(cases), settings.eval_data_file.name)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures: list[str] = []

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=settings.project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client,
    ):
        judge = settings.judge_deployment or resolve_deployment(project_client, settings.model_name)
        log.info("Judge model deployment: %s", judge)

        for agent_name in [args.agent] if args.agent else settings.agent_names:
            agent = agent_runner.get_agent_info(project_client, agent_name, args.agent_version or settings.agent_version)
            log.info("Evaluating agent %s v%s in '%s' mode", agent.name, agent.version, args.mode)

            run_eval = evaluate_simulation if args.mode == "simulate" else evaluate_dataset
            results = run_eval(args, settings, project_client, openai_client, agent, judge, cases, stamp, failures)
            if results is None:
                continue

            path = evaluation.save_results(results, settings.results_dir, f"{agent.name}-{args.mode}")
            evaluation.print_summary(results)
            log.info("Saved results to %s", path)
            failures.extend(f"{agent.name}: {f}" for f in evaluation.gate(results, gate_threshold))

    if failures:
        print("\nQUALITY GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

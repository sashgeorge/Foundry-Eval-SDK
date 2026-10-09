"""Configuration loading.

Values come from the project `.env` file, and real environment variables win so the
same code runs unchanged in CI/CD. Both the hyphenated keys used in this repo's `.env`
(e.g. `FOUNDRY-PROJECT`) and the conventional SDK names (e.g. `FOUNDRY_PROJECT_ENDPOINT`)
are accepted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _lookup(values: dict[str, str | None], *keys: str, default: str | None = None) -> str | None:
    # Real environment variables (any alias) take precedence over .env values.
    for source in (os.environ, values):
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


@dataclass(frozen=True)
class Settings:
    project_endpoint: str
    agent_names: list[str]
    model_name: str
    eval_data_file: Path
    agent_version: str | None = None
    judge_deployment: str | None = None
    simulator_deployment: str | None = None
    criteria_threshold: float = 0.8
    pass_rate_gate: float = 0.8
    output_dir: Path = field(default=PROJECT_ROOT / ".foundry")

    @property
    def datasets_dir(self) -> Path:
        return self.output_dir / "datasets"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"


def load_settings() -> Settings:
    values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}

    endpoint = _lookup(values, "FOUNDRY-PROJECT", "FOUNDRY_PROJECT_ENDPOINT", "AZURE_AI_PROJECT_ENDPOINT")
    agents = _lookup(values, "AGENT-NAMES", "FOUNDRY_AGENT_NAME", "AGENT_NAMES")
    model = _lookup(values, "MODEL-NAME", "FOUNDRY_MODEL_NAME", "AZURE_AI_MODEL_DEPLOYMENT_NAME")
    data_file = _lookup(values, "EVAL-DATA-FILE", "EVAL_DATA_FILE")

    missing = [
        name
        for name, value in {
            "FOUNDRY-PROJECT": endpoint,
            "AGENT-NAMES": agents,
            "MODEL-NAME": model,
            "EVAL-DATA-FILE": data_file,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required settings in {ENV_FILE}: {', '.join(missing)}")

    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    return Settings(
        project_endpoint=endpoint,
        agent_names=[a.strip() for a in agents.split(",") if a.strip()],
        model_name=model,
        eval_data_file=data_path,
        agent_version=_lookup(values, "AGENT-VERSION", "FOUNDRY_AGENT_VERSION"),
        judge_deployment=_lookup(values, "EVAL-MODEL-DEPLOYMENT", "EVAL_MODEL_DEPLOYMENT"),
        simulator_deployment=_lookup(values, "SIMULATOR-MODEL-DEPLOYMENT", "SIMULATOR_MODEL_DEPLOYMENT"),
        criteria_threshold=float(_lookup(values, "CRITERIA-THRESHOLD", "CRITERIA_THRESHOLD", default="0.8")),
        pass_rate_gate=float(_lookup(values, "PASS-RATE-GATE", "PASS_RATE_GATE", default="0.8")),
    )


def resolve_deployment(project_client, model_or_deployment: str) -> str:
    """Return a deployment name for `model_or_deployment`.

    Evaluators need a *deployment* name. `.env` holds a model name (e.g. `gpt-5.6-sol`)
    whose deployment may be named differently (e.g. `gpt-5.6-sol-1`), so map it here.
    """
    deployments = list(project_client.deployments.list())
    names = {d.name for d in deployments}
    if model_or_deployment in names:
        return model_or_deployment
    matches = sorted(d.name for d in deployments if getattr(d, "model_name", None) == model_or_deployment)
    if not matches:
        raise SystemExit(
            f"No deployment found for model '{model_or_deployment}'. "
            f"Available deployments: {', '.join(sorted(names))}"
        )
    return matches[0]

# Multi-turn Agent Evaluation (Microsoft Foundry SDK)

Evaluates a Foundry agent on **multi-turn conversations** using the Foundry Evaluation SDK
(`azure-ai-projects` v2 + OpenAI-compatible `evals` API). Scores each conversation as a whole
(conversation-level), grades tool use, and checks each scenario's own `expected_outcome` /
`evaluation_criteria`.

Reference: [Evaluate your AI agents](https://learn.microsoft.com/azure/foundry/observability/how-to/evaluate-agent?tabs=python)
and its sub-pages: [conversation datasets](https://learn.microsoft.com/azure/foundry/observability/how-to/cloud-evaluation-conversations),
[simulate conversations](https://learn.microsoft.com/azure/foundry/observability/how-to/cloud-evaluation-simulate-conversations),
[agent evaluators](https://learn.microsoft.com/azure/foundry/concepts/evaluation-evaluators/agent-evaluators),
[dataset schema](https://learn.microsoft.com/azure/foundry/observability/how-to/evaluation-dataset-schema),
[custom evaluators](https://learn.microsoft.com/azure/foundry/concepts/evaluation-evaluators/custom-evaluators).

---

## 1. Quick start

```powershell
pip install -r requirements.txt
az login                                   # DefaultAzureCredential uses your Azure CLI login
python run_evaluation.py                   # replay mode (default) against the live agent
python run_evaluation.py --mode recorded   # score the stored conversations as-is
python run_evaluation.py --mode simulate   # Foundry simulates the user from test_case_description
```

Required role: **Foundry User** (formerly Azure AI User) on the Foundry project.

### Configuration (`.env`)

| Key | Required | Purpose |
|---|---|---|
| `FOUNDRY-PROJECT` | yes | Project endpoint `https://<account>.services.ai.azure.com/api/projects/<project>` |
| `AGENT-NAMES` | yes | Agent(s) to evaluate; comma-separated for several |
| `MODEL-NAME` | yes | Judge model. A model name (`gpt-5.6-sol`) is automatically mapped to its deployment (`gpt-5.6-sol-1`) |
| `EVAL-DATA-FILE` | yes | Evaluation data (`examples/multiturn-example.jsonl` is a fictional sample) |
| `AGENT-VERSION` | no | Pin an agent version (default: latest) |
| `EVAL-MODEL-DEPLOYMENT` | no | Explicit judge deployment; skips the model-to-deployment lookup |
| `SIMULATOR-MODEL-DEPLOYMENT` | no | Model that plays the user in `simulate` mode (default: judge) |
| `CRITERIA-THRESHOLD` | no | Pass threshold for the scenario criteria evaluator, 0–1 (default `0.8`) |
| `PASS-RATE-GATE` | no | Minimum pass rate per evaluator for the quality gate (default `0.8`) |

Real environment variables override `.env`, and the standard SDK names also work
(`FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_AGENT_NAME`, `FOUNDRY_MODEL_NAME`, `EVAL_DATA_FILE`), so the same code runs in CI/CD.

### Synthetic multi-turn data

[synthetic_data/](synthetic_data/) contains a generator that creates an evaluation file with messages **and tool calls**
by running the live agent with its own prompt and tools against a simulated caller. Each conversation starts with
the agent asking for the caller's name or policy number. Generated healthcare conversations may contain sensitive
identifiers; review and sanitize them before sharing or committing.

```powershell
Copy-Item .env.example .env
# Edit .env with your Foundry project, agent, model, and dataset values.
python run_evaluation.py --mode recorded
```

If a data file includes `tool_definitions`, `recorded` mode also runs `tool_use_quality`.

---

## 2. How it works

```text
EvalData-*.jsonl ──► dataset.py (validate + reshape)
                        │
      ┌─────────────────┼──────────────────────────┐
  recorded            replay                    simulate
  stored messages     agent_runner.py replays   seeds → Foundry simulated
                      user turns on live agent   user ↔ live agent
                      (captures MCP tool calls)
      └────────┬────────┘                          │
     messages-schema dataset (content-hash version) │
               │                                   │
  ┌────────────┴─────────────┐                     │
  Eval A: conversation quality   Eval B: scenario criteria
  (evaluation_level=conversation) (custom judge, per row)   Eval A only
  └────────────┬─────────────┘                     │
               ▼                                   ▼
     evaluation.py: poll → collect → merge → .foundry/results/*.json + console + quality gate
```

| Module | Responsibility |
|---|---|
| [config.py](multiturn_eval/config.py) | Load `.env` + env vars, resolve the judge deployment |
| [dataset.py](multiturn_eval/dataset.py) | Validate source rows, build `messages` rows / simulation seeds, upload versioned datasets |
| [agent_runner.py](multiturn_eval/agent_runner.py) | Replay user turns in a server-side conversation; convert output to the eval message schema |
| [evaluators.py](multiturn_eval/evaluators.py) | Built-in testing criteria + versioned custom scenario evaluator |
| [evaluation.py](multiturn_eval/evaluation.py) | Create or reuse evaluations, start runs, poll, aggregate results, quality gate |
| [run_evaluation.py](run_evaluation.py) | CLI entry point |

### Modes – which one to use

| Mode | What is scored | Use when |
|---|---|---|
| `replay` (default) | The **live agent's** responses to the scripted user turns, including tool calls | Regression testing of agent versions/prompts; deterministic user script |
| `recorded` | The conversation exactly as stored in the file | Scoring production transcripts, human-written golden conversations, or logs |
| `simulate` | Conversations generated by a simulated user from `test_case_description` | Coverage and robustness; the user adapts to what the agent says |

> Replay limitation: scripted user turns don't adapt. If the agent asks something the script
> doesn't answer, the conversation can drift. Use `simulate` to cover that case.

### Evaluators

**Eval A – conversation quality** (built-in, `evaluation_level="conversation"`, mapped from `{{item.messages}}`):

| Evaluator | Measures |
|---|---|
| `builtin.output_quality` | Composite: fluency, coherence, intent resolution, task adherence, groundedness, task completion (one LLM call) |
| `builtin.customer_satisfaction` | Helpfulness, completeness, clarity, tone, resolution, and adaptability across the whole conversation (1–5) |
| `builtin.task_completion` | Did the agent complete what the user asked, end to end |
| `builtin.tool_use_quality` | Composite: tool call accuracy/success, input accuracy, output utilization, selection. Only when tool calls exist (`replay`) |

**Eval B – scenario criteria** (custom prompt evaluator `scenario-criteria-adherence`): grades each
conversation against **its own** `expected_outcome` and `evaluation_criteria`. Result = fraction of
criteria met (0–1). Includes order-sensitive checks such as "verify identity before accessing claims".

Composite evaluators report a primary result plus component metrics. The summary shows both
(for example `output_quality` and `output_quality.groundedness`). The quality gate only uses primary results.

### Outputs

* Console summary: pass/fail per evaluator and component, failure reasons, portal report links
* `.foundry/results/<agent>-<mode>-<timestamp>.json`: merged results plus raw output items
* `.foundry/datasets/*.jsonl`: the exact data uploaded (useful for debugging data mappings)
* Exit code `1` when the quality gate fails (CI/CD-friendly)

---

## 3. Evaluation SDK usage guidance

### 3.1 Clients
```python
with (
    DefaultAzureCredential() as credential,
    AIProjectClient(endpoint=endpoint, credential=credential) as project_client,
    project_client.get_openai_client() as openai_client,   # evals + responses + conversations
):
```
`project_client` handles Foundry assets (datasets, evaluators, agents, deployments).
`openai_client` handles `evals`, `evals.runs`, `responses`, and `conversations`.
Don't use the v1 `AIProjectClient.from_connection_string()` API.

### 3.2 Multi-turn data: the `messages` schema
One row = one conversation. Tool calls and results go **inside** `content`:
```json
{"messages": [
  {"role": "system", "content": "<agent instructions>"},
  {"role": "user", "content": "I received an EOB..."},
  {"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "c1", "name": "lookup", "arguments": {"q": "Jim"}}]},
  {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": "..."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "Thanks, Jim..."}]}
 ],
 "tool_definitions": [{"name": "lookup", "description": "...", "parameters": {"type": "object", "properties": {}}}]}
```
Include the agent's system prompt so `task_adherence` can judge against the actual instructions.

### 3.3 Conversation-level run
```python
openai_client.evals.runs.create(
    eval_id=eval_id, name=run_name,
    data_source={"type": "jsonl", "source": {"type": "file_id", "id": dataset_id}},
    extra_body={"evaluation_level": "conversation"},   # default is "turn" (last response only)
)
```
Only some evaluators support conversation level: `output_quality`, `tool_use_quality`,
`customer_satisfaction`, `task_completion`, `coherence`, `groundedness`, `deflection_rate`.
Turn-only evaluators (for example `fluency`, `relevance`, or the safety evaluators) are **skipped**, not failed.

### 3.4 Live agent invocation (replay)
```python
conv = openai_client.conversations.create()
for user_text in user_turns:
    r = openai_client.responses.create(
        conversation=conv.id, input=user_text,
        extra_body={"agent_reference": {"name": agent, "version": version, "type": "agent_reference"}},
    )
    # r.output contains: mcp_list_tools, mcp_call / function_call, message, reasoning
```

### 3.5 Simulation
```python
data_source={
  "type": "azure_ai_user_conversation_simulation",
  "source": {"type": "file_id", "id": seeds_id},       # test_case_id, test_case_description, simulation_configuration
  "target": {"type": "azure_ai_agent", "name": agent, "version": version},
  "model_configuration": {"model": simulator_deployment},
  "default_simulation_configuration": {"max_num_turns": 12, "conversation_repetitions": 1,
                                       "enable_conversation_dataset_generation": True},
}
```

### 3.6 Custom evaluators
```python
project_client.evaluators.create_version(
    name=..., evaluator_version={..., "supported_evaluation_levels": ["turn"], "definition": {...}},
    headers={"Foundry-Features": "Evaluations=V1Preview"},   # preview opt-in is required
)
```
Reference it in testing criteria with `evaluator_name` **and** `evaluator_version`.

---

## 4. Best practices

**Evaluation design**
1. **Score the whole conversation.** Turn-level scoring only judges the last reply and misses
   things like repeated failed actions, skipped verification, or user frustration. Use `evaluation_level="conversation"`.
2. **Layer evaluators.** Use built-in composites for general quality and tool use, plus one custom
   judge for business rules that differ per scenario. Prefer composites (`output_quality`,
   `tool_use_quality`) over many separate evaluators to cut cost and latency.
3. **Write per-scenario criteria that can be checked.** For example: "Requests the user's name or
   policy number *before* accessing claim details". Avoid vague criteria such as "is helpful".
4. **Evaluate the live agent, not only transcripts.** `recorded` grades a transcript. `replay` and
   `simulate` grade the agent, and they're the modes that catch regressions from prompt, tool, or model changes.
5. **Include tool definitions and the system prompt** so the judge can check tool choice and policy adherence.
6. **Add safety coverage separately.** Content-safety evaluators are turn-level, so run them in a
   turn-level run or through continuous evaluation in production.

**Reproducibility and comparison**
7. **Pin everything.** That includes the agent version (`--agent-version`), the evaluator version
   (pinned automatically), and the dataset version (content hash). Every run records them in its metadata.
8. **Reuse one evaluation per configuration** so runs line up in the portal for comparison across agent
   versions. The code reuses an evaluation when its name and criteria fingerprint match.
9. **Account for non-determinism.** LLM agents and judges vary between runs. For release decisions,
   use more test cases, `--repetitions > 1` in simulate mode, and look at pass rates instead of single scores.

**Judge model**
10. Use a strong judge from a different model family or tier than the agent when possible. Composite
    evaluators recommend the GPT-5.6 family. `gpt-5.6-luna` is the lowest-cost option that keeps quality high.
11. Reasoning models (gpt-5.x, o-series) **reject `temperature`/`top_p`**. Leave sampling parameters unset for them.

**Data**
12. Validate data before uploading (roles, at least one user turn, unique IDs). This code fails fast with line numbers.
13. Truncate large tool outputs (retrieval payloads) to control judge cost and context size. The default is 4,000 characters.
14. Healthcare/PII: use synthetic identities in test data. Replay conversations are deleted by default (`--keep-conversations` to retain them).
15. Grow the dataset from real traffic: [traces → datasets](https://learn.microsoft.com/azure/foundry/observability/how-to/traces-to-dataset)
    and [synthetic seeds](https://learn.microsoft.com/azure/foundry/observability/how-to/evaluation-dataset-synthetic).

**Operations**
16. **Make it a CI/CD gate.** `python run_evaluation.py --fail-under 0.8` returns a non-zero exit code on regression.
    Use a federated (OIDC) service principal with the Foundry User role, not secrets.
17. Retry transient errors with backoff (built into replay), isolate failures per test case, and set run timeouts.
18. Add [continuous evaluation](https://learn.microsoft.com/azure/foundry/observability/how-to/how-to-monitor-agents-dashboard)
    for production traffic. Offline evaluation doesn't replace production monitoring.

---

## 5. Troubleshooting (verified behaviors)

| Symptom | Cause / fix |
|---|---|
| `requires opt-in preview feature Evaluations=V1Preview` | Pass `headers={"Foundry-Features": "Evaluations=V1Preview"}` to `evaluators.*` calls |
| `Evaluator ... does not support conversation-level evaluation and was skipped` | Custom prompt evaluators default to `supported_evaluation_levels=["turn"]` |
| `Conversation-level evaluation requires 'messages' in data_mapping` | Every evaluator in a conversation-level run must map `messages` |
| Custom prompt placeholders arrive empty in a conversation-level run | At conversation level only `{{messages}}` is injected into custom prompts. Per-row fields need a turn-level run (Eval B) |
| `Unsupported parameter: 'temperature'` during simulation | Reasoning simulator model; don't send `sampling_params` (or use `--simulator-temperature` with a non-reasoning model) |
| `Simulation produced no valid conversations` / `Agent returned empty response` | Usually transient on single-seed runs; re-run. With more seeds, isolated failures don't fail the run |
| `No deployment found for model` | `MODEL-NAME` must match a deployment or a deployed model name; or set `EVAL-MODEL-DEPLOYMENT` |
| Evaluator `not graded` in the quality gate | The evaluator was skipped or errored. Check its level support and data mappings |

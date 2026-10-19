# RoadSense architecture

## What it is

A question about a road-scene photograph goes in. The agent decides what evidence it
needs, gathers it with vision tools, decides again in light of what came back, and
answers using only what it actually found — or says it cannot.

**What it is not:** a safety system. RoadSense reads static image files. It does not
touch a vehicle, a camera, CAN/ECU/ADAS, or SYNC, and it never advises on braking,
collision risk, or what a driver should do. The demonstrated value is the decision
pattern, the traceability, and the measurements.

## Components

| Component | File | Responsibility |
|---|---|---|
| CLI | `cli.py` | one question + one image → printed trace + answer |
| Loop | `pattern/evidence_seeking.py` | the reusable engine: ask, veto, execute, stop |
| Domain wiring | `agent.py` | the four actions, how to run them, the trace |
| State | `state.py` | accumulated evidence, and its text rendering |
| Action contract | `decision_schema.py` | the closed action set |
| Guardrails | `policy.py` | legality, masking, override fallback |
| Decider client | `decider_client.py` | in-process or HTTP Strands Decider 2B |
| Vision tools | `tools/` | Rekognition DetectLabels / DetectText |
| Writer | `bedrock_client.py` | Bedrock Converse, evidence-only prompt |
| Telemetry | `telemetry.py` | JSON trace, token counts, cost |

## Data flow

```
question + image path
        │
        ▼
   AgentState ──render_for_decider()──▶ text  ──┐
        ▲                                        │
        │                             ┌──────────▼────────────┐
        │                             │  Strands Decider 2B   │  (EC2, CPU)
        │                             │  choice + noul        │
        │                             └──────────┬────────────┘
        │                                        │ probabilities, confidence
        │                             ┌──────────▼────────────┐
        │                             │  policy.legality      │
        │                             └──────────┬────────────┘
        │                                        ▼
        │                        ┌───────────────────────────────┐
        └────── Evidence ────────│ Rekognition DetectLabels/Text │
                                 └───────────────────────────────┘
                                                 │
                                   (loop until terminal or budget)
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │ Bedrock Converse (Haiku 4.5)  │
                                 │ sees evidence text, NOT image │
                                 └───────────────┬───────────────┘
                                                 ▼
                                        answer + JSON trace
```

Two separations carry most of the design:

**Decider never sees the image.** It reasons over an evidence summary. That is what
makes the selector reusable outside vision, and what lets the decision layer be
tested with a string.

**Bedrock never sees the image either.** The writer cannot describe what the tools
did not report, so "grounded in the evidence" is a property of the plumbing rather
than a promise in a prompt.

## The Decider 2B contract

Read off the installed package (`strands-decider 0.1.0`), not from documentation —
published write-ups disagreed on model IDs and none documented the Python API.

```python
from strands_decider.infer import load_engine          # infer.py:457
from strands_decider.schema import ChoiceQuestion, NoulQuestion

engine = load_engine(checkpoint, device="cpu")          # defaults to "cuda"; fails on CPU hosts
resp = engine.ask(state=<str|dict|list>, questions={...})

resp.answers["next_action"].choice          # str, one of the option keys
resp.answers["next_action"].probabilities   # dict[str, float]
resp.answers["next_action"].confidence      # (N*p_max - 1)/(N - 1), calibrated
resp.answers["evidence_sufficient"].noul    # P(true)
resp.usage.input_tokens / .output_tokens
```

`ChoiceQuestion` takes **`criteria`** (a `{name: description}` dict), not `options`.
Both questions go in one `ask`, so the state is encoded once and its KV cache is
broadcast across the question batch.

The model is a LoRA adapter (202 MB) + pointer head (12.6 MB) over
`Qwen/Qwen3.5-2B-Base` (4.56 GB). It is **not** a Bedrock model; it runs on compute
you manage.

## Deployment

| Resource | Identifier | Notes |
|---|---|---|
| EC2 | `c7i.2xlarge`, Ubuntu 24.04, 40 GB gp3 | hosts the Decider weights; ~$0.36/h |
| IAM role | `roadsense-ec2-role` | instance profile — no user keys on the box |
| S3 | `roadsense-demo-521106215898` | traces and assets; public access blocked |
| Bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | us-east-1 inference profile |
| Rekognition | `DetectLabels`, `DetectText` | us-east-1 |

Minimum IAM for the agent: `bedrock:InvokeModel` + `bedrock:Converse` on
foundation-model and inference-profile ARNs; `rekognition:DetectLabels` and
`rekognition:DetectText`; `s3:GetObject`/`PutObject` scoped to the project bucket.

The GPU path is untested here: 4 GB of VRAM on the development laptop could not hold
a 2B model comfortably, so every reported latency is CPU.

## Measured characteristics

| | Value | How it was obtained |
|---|---|---|
| Decider model load | 39.8 s | once per process, EC2 CPU |
| Decider per decision | ~2.6–3.7 s | two questions per call |
| Rekognition per call | 0.14–1.3 s | real calls, us-east-1 |
| Decider tokens | ~290–480 in, 2 out | `resp.usage` |

`transformers` reports that `causal_conv1d` and `flash-linear-attention` are not
installed and falls back to reference kernels, so the latency above is an upper
bound rather than the model's best.

## Safety and privacy

- No credentials in the repo, in logs, or on the instance — the EC2 box authenticates
  with an instance profile.
- Tool payloads are kept in the trace for auditability; the demo images are synthetic
  and contain no people, faces, or plates.
- Face and plate recognition are deliberately not used. `DetectLabels` and
  `DetectText` are the only Rekognition operations the IAM policy allows.

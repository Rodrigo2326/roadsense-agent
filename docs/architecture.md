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
        │                             │  Strands Decider 2B   │  (EC2, GPU)
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
| EC2 (CPU) | `c7i.2xlarge`, Ubuntu 24.04 | reproducible without a GPU quota; ~$0.36/h |
| EC2 (GPU) | `g5.xlarge`, NVIDIA A10G 24 GB | where the latency numbers come from; ~$1.01/h |
| IAM role | `roadsense-ec2-role` | instance profile — no user keys on the box |
| S3 | `roadsense-demo-<account-id>` | traces and assets; public access blocked |
| Bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | us-east-1 inference profile |
| Rekognition | `DetectLabels`, `DetectText` | us-east-1 |

Minimum IAM for the agent: `bedrock:InvokeModel` + `bedrock:Converse` on
foundation-model and inference-profile ARNs; `rekognition:DetectLabels` and
`rekognition:DetectText`; `s3:GetObject`/`PutObject` scoped to the project bucket.

Running on a GPU needs an `All G and VT instances` service quota above zero; it is
zero by default on a new account and takes a support case to raise.

## Measured characteristics

| | CPU (c7i.2xlarge) | GPU (A10G) | How it was obtained |
|---|---|---|---|
| Model load | 39.8 s | 10–45 s | once per process |
| Per decision, 1 question | 1 278 ms | **78 ms** | `scripts/profile_latency.py`, 303-char state |
| Per decision, 2 questions | 2 241 ms | **163 ms** | the configuration the agent ships with |
| Rekognition per call | 0.14–1.3 s | same | real calls, us-east-1 |
| Decider tokens | ~290–480 in, 2 out | same | `resp.usage` |

Three things explain the gap, and only one of them is a setting:

- **dtype.** On CUDA the torso stays in bfloat16; on CPU the package deliberately
  upcasts it to fp32, because CPU bf16 kernels are slower than fp32 ones.
- **Kernels.** `causal-conv1d` is a source-only CUDA extension and
  `flash-linear-attention` is Triton, so on a CPU host the fallback warnings
  `transformers` prints are accurate *and* unfixable. On GPU,
  `flash-linear-attention` installs cleanly and is worth 20–31 %; `causal-conv1d`
  did not finish compiling, so there is headroom still unclaimed.
- **The second question.** It costs ~0.9 s on CPU and ~85 ms on GPU, and nothing in
  the control flow reads it — see `docs/findings.md`.

For reference, the package's own source quotes 3.5 s for a 256-token question in fp32
on an M3 Pro, so the CPU column is in line with what its authors measured.

## Safety and privacy

- No credentials in the repo, in logs, or on the instance — the EC2 box authenticates
  with an instance profile.
- Tool payloads are kept in the trace for auditability.
- The benchmark images are synthetic and contain no people, faces or plates. The demo
  also ships four Wikimedia Commons photographs (three CC0, one CC BY), and those are
  real traffic: Rekognition read two number plates off the street scene on the first
  try. `scripts/fetch_commons_assets.py` blurs every text region it can find in that
  file and re-reads it afterwards, failing if any word survives. Attribution and what
  was changed are in `demo/photos/ATTRIBUTION.md`.
- Face and plate recognition are deliberately not used. `DetectLabels` and
  `DetectText` are the only Rekognition operations the IAM policy allows.

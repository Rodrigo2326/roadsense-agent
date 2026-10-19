# RoadSense — an evidence-seeking agent for visual decisions

> Instead of guessing from one image, RoadSense decides what evidence to gather next,
> shows its reasoning path, and knows when the available evidence is not enough.

Built for the **AWS Agent Experience Hackathon: Strands Decider**.
Tags: `#strands-decider-2b` `#tool-selection` · Licence: Apache-2.0

**Strands Decider 2B** chooses which vision tool to run next from a typed, closed
action set. **Amazon Rekognition** gathers the evidence. **Amazon Bedrock** writes an
answer limited to what was actually found — it never sees the image, so it cannot
describe what the tools did not report.

The reusable half is the
[**Evidence-Seeking Tool Selection**](docs/decision-pattern.md) pattern, which has no
road, image or AWS call in it: [`src/roadsense/pattern/`](src/roadsense/pattern/).

**What this is not:** a safety system. RoadSense reads static image files. It does not
touch a vehicle, a camera, CAN/ECU/ADAS or SYNC, and it never advises on braking,
collision risk, or what a driver should do.

---

## What a run looks like

```
$ roadsense -q "What vehicles appear and what text can be read on the sign?" \
            -i demo/images/synthetic_sign_clear.jpg

=== decision trace ===
  step 1: detect_objects   confidence=0.776  P(evidence sufficient)=0.196  3325ms
      detect_objects=0.832  read_sign_text=0.158  request_better_evidence=0.006  answer=0.004
  step 2: read_sign_text   confidence=0.952  P(evidence sufficient)=0.273  3532ms
      read_sign_text=0.964  detect_objects=0.026  answer=0.006  request_better_evidence=0.004
  step 3: answer           confidence=0.501  P(evidence sufficient)=0.572  3719ms

=== evidence ===
  detect_objects: ok=True confidence=1.00 (241ms)
      Objects detected: Road (1.00), Sign (1.00), Symbol (1.00), Bus Stop (0.93)...
  read_sign_text: ok=True confidence=1.00 (468ms)
      OCR read the text: 'STOP | AHEAD'.

=== answer (answered) ===
No vehicles appear in the image. The sign reads "STOP | AHEAD" with high confidence.
```

The second decision is made *after* seeing the first tool's output, and changes
because of it. `P(evidence sufficient)` rises 0.196 → 0.273 → 0.572 as evidence
accumulates. Every number above is from a real run.

### When the evidence is not enough

```
$ roadsense -q "What text can be read on the sign?" \
            -i demo/images/synthetic_sign_illegible.jpg
=== answer (answered) ===
No readable text could be extracted from the sign in this image. The OCR tool found
no legible characters, though object detection confirmed the presence of a sign.

$ roadsense -q "What vehicles appear in this photo?"      # no image at all
  step 1: request_better_evidence  confidence=0.924
=== answer (insufficient_evidence) ===
No image was provided to RoadSense, so no vehicles can be identified.
```

---

## Setup

Decider 2B is **not a Bedrock model**. It is a LoRA adapter (202 MB) plus a pointer
head (12.6 MB) over `Qwen/Qwen3.5-2B-Base` (4.56 GB), and it runs on compute you
manage. Budget ~8 GB of disk for the weights and the PyTorch stack.

```bash
python -m venv .venv && . .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"                          # the agent
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[decider]"                      # strands-decider + weights
cp .env.example .env                             # no secrets go in here
```

AWS credentials come from the usual chain — `~/.aws/credentials` locally, an
instance profile on EC2. **Never put keys in `.env` or the repo.**

### Minimum IAM

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow",
   "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
   "Resource": ["arn:aws:bedrock:*::foundation-model/*",
                "arn:aws:bedrock:*:<account>:inference-profile/*"]},
  {"Effect": "Allow",
   "Action": ["rekognition:DetectLabels", "rekognition:DetectText"],
   "Resource": "*"}
]}
```

Face and plate recognition are deliberately absent: `DetectLabels` and `DetectText`
are the only Rekognition operations the agent can call.

---

## Running the demo from scratch

```bash
# 1. Generate the image set (deterministic, Apache-2.0, no faces or plates)
python scripts/make_assets.py

# 2. Verify the live integrations before trusting anything
python scripts/smoke_aws.py          # Rekognition + Bedrock, real calls
python scripts/smoke_decider.py      # Strands Decider 2B, 3 real decisions

# 3. The headline case: a question that needs two tools, in sequence
python -m roadsense.cli \
  -q "What vehicles appear and what text can be read on the sign?" \
  -i demo/images/synthetic_sign_clear.jpg \
  --trace trace_clear.json

# 4. The honest-limits case: OCR returns nothing, the agent does not invent text
python -m roadsense.cli -q "What text can be read on the sign?" \
  -i demo/images/synthetic_sign_illegible.jpg

# 5. The no-evidence case: nothing can be gathered at all
python -m roadsense.cli -q "What vehicles appear in this photo?"

# 6. The pattern, with no image in it
python examples/triage_example.py

# 7. Tests
python -m pytest tests/ -q
```

### Driving a remote decider

The 4.5 GB of weights do not have to live on the machine running the agent:

```bash
# on the box with the weights
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000

# on your laptop
python -m roadsense.cli -q "..." -i image.jpg --decider-endpoint http://host:8000
```

---

## Benchmark

```bash
python eval/make_cases.py                                   # regenerate cases.jsonl
python eval/run_benchmark.py --arms keyword exhaustive llm decider --tag full
```

37 cases across easy / medium / hard / invalid-input, 19 answerable and 18 where the
right move is to say so. Four selectors run through the *same* loop, the same tools
and the same writer — only the thing choosing the next action changes:

| Arm | What chooses the tool |
|---|---|
| `keyword` | an if-statement over the question. The honest bar to clear. |
| `exhaustive` | run every legal tool, then answer. Never misses; pays every time. |
| `llm` | Bedrock picks from the same menu. |
| `decider` | Strands Decider 2B. |

Results land in [`eval/results/`](eval/results/), written by the run and never typed
by hand. Every metric is deterministic and reads only the trace; no model judges
another model's output.

---

## Reading the trace

```jsonc
{
  "outcome": "answered",              // or insufficient_evidence / error
  "stop_reason": "terminal_action",   // or budget_exhausted
  "policy_overrides": 0,              // times legality overruled the model
  "steps": [{
    "action": "read_sign_text",
    "confidence": 0.952,              // Decider's calibrated confidence
    "probabilities": {...},           // the full distribution it scored
    "evidence_sufficient": 0.273,     // P(true) from the paired yes/no question
    "offered_actions": [...],         // what was on the menu at this step
    "decider_called": true,           // false on a budget-forced step
    "rejected": null                  // why a pick was overruled, if it was
  }],
  "evidence": [...],                  // normalised + raw tool payloads
  "cost": {...}                       // real counts; dollars only where verified
}
```

`confidence` is `null` on a forced step. Nothing decided it, and writing `0.0` there
would read like a decision that happened.

---

## Honest limitations

- **Latency is the headline cost.** ~2.6–3.7 s per decision on CPU, plus a 39.8 s
  one-off model load. `transformers` reports that `causal_conv1d` and
  `flash-linear-attention` are missing and falls back to reference kernels, so this
  is an upper bound rather than the model's best. The GPU path is untested — 4 GB of
  VRAM on the development laptop could not hold the model comfortably.
- **Decider repeats tools whose results it already has** when all four actions are
  offered. Masking illegal actions out of the menu fixes this and *raises* its
  confidence; the benchmark runs both arms, and `--no-mask` reproduces the worse one.
- **Rekognition's OCR confidence is bimodal on this font.** It reads the word at
  ~0.98, reads something else at ~0.85, or returns nothing. A confidence threshold
  alone cannot separate right from wrong here — the `garbled` case clears any sane
  floor while being wrong. Reported rather than tuned away.
- **The detector does not recognise the drawn vehicles** in the synthetic images as
  vehicles; it returns a `Vehicle` category at 0.60 with zero instances. The cases
  that ask about vehicles are therefore scored as *insufficient*, which is what the
  evidence honestly supports.
- **The fidelity check is narrow.** It catches quoted text no tool produced. It
  cannot catch every unsupported claim in prose, and says so rather than implying a
  completeness it does not have.
- **Bedrock cost is reported as tokens, not dollars.** No rate for
  `claude-haiku-4-5` could be verified from the AWS Pricing API or the public pricing
  page. Rekognition's $0.001/image *is* verified and is priced.
- **Synthetic images only.** The licence is unambiguous and the degradation is a
  measured dial rather than an accident, but they are not photographs.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Torch not compiled with CUDA enabled` | `load_engine` defaults to `device="cuda"`. Pass `--decider-device cpu`. |
| `decider unavailable: strands-decider is not installed` | `pip install -e ".[decider]"`, or use `--decider-endpoint`. |
| `AccessDeniedException` on `converse` | the model id must be an **inference profile** (`us.anthropic...`), and your account needs access to it. |
| First run hangs for a minute | it is downloading 4.5 GB of weights, then loading for ~40 s. |
| `ValidationException` from Rekognition | the file is not a real JPEG/PNG — the agent reports this as unusable evidence rather than crashing. |

---

## Layout

```
src/roadsense/
  pattern/evidence_seeking.py   the reusable pattern — no domain in it
  agent.py                      the four actions, wired into the pattern
  decider_client.py             Strands Decider 2B: in-process or HTTP
  policy.py                     legality, masking, override fallback
  state.py                      evidence, and its text rendering
  tools/                        Rekognition DetectLabels / DetectText
  bedrock_client.py             the writer, evidence-only
eval/                           cases, baselines, metrics, runner
examples/triage_example.py      the pattern with no image in it
docs/                           decision-pattern.md, architecture.md
```

## Licence

Apache-2.0. Demo images are generated by `scripts/make_assets.py` and are original
work under the same licence; see [`demo/assets_manifest.md`](demo/assets_manifest.md).

# Most agents decide once. This one decides, looks, and decides again.

**Tags:** `#strands-decider-2b` `#tool-selection`
**Repo:** https://github.com/<your-handle>/roadsense-agent · Apache-2.0

Ask an agent a question about a photograph and it usually does one of two things:
calls every tool it has, or picks one tool from the wording of the question and
commits. Neither is a decision. The first is a shopping list; the second is a regex
with extra steps.

RoadSense does something else. It has four actions, it picks one, it looks at what
came back, and then it picks again — and the second pick is frequently not the one it
would have made first. **Strands Decider 2B** makes those choices. Amazon Rekognition
gathers the evidence. Amazon Bedrock writes the final answer and never sees the image,
so it cannot describe anything the tools did not report.

Everything below is measured. The numbers come from 37 cases run through four
different selectors in the same loop, and the losses are reported alongside the wins.

---

## The problem: you don't know which evidence you need

"What vehicles appear and what text can be read on the sign?"

That question needs two different tools. "What does the sign say?" needs one. "Are
there any people here?" needs the other. And a question that needs OCR is worthless if
the sign turns out to be unreadable — at which point the honest move is to say so,
not to guess a word.

You cannot know up front which of those situations you are in. You have to go and look.

So the agent gets four actions and nothing else:

| action | what it does |
|---|---|
| `detect_objects` | Rekognition `DetectLabels` |
| `read_sign_text` | Rekognition `DetectText` |
| `request_better_evidence` | stop, and say what is missing |
| `answer` | stop, and answer from what was gathered |

## Why Strands Decider 2B is the right model for this

Decider 2B is a **pointer** model. It does not generate text; it scores a fixed list
of options and returns a distribution with a calibrated confidence. That changes what
can go wrong.

A generative model asked to "pick a tool" can emit a tool that does not exist, emit
malformed arguments, or quietly narrate a tool call it never made. None of those are
failure modes here — not because we validate them away, but because the architecture
has no slot for them. The model physically cannot return a name that is not a key of
the menu.

The API is two questions in one call:

```python
from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion, NoulQuestion

engine = load_engine(checkpoint, device="cuda")

resp = engine.ask(state=state_text, questions={
    "next_action": ChoiceQuestion(
        instructions="Which single action should the agent take next to answer the "
                     "question with evidence it can actually verify?",
        criteria=menu,                       # {name: description}
    ),
    "evidence_sufficient": NoulQuestion(
        instructions="Is the evidence collected so far already sufficient?"),
})

resp.answers["next_action"].choice          # "read_sign_text"
resp.answers["next_action"].probabilities   # the full distribution
resp.answers["next_action"].confidence      # (N·p_max − 1)/(N − 1), calibrated
```

Two notes for anyone wiring this up. `ChoiceQuestion` takes **`criteria`**, not
`options` — I read that off `strands_decider/schema.py` after published write-ups
disagreed. And `load_engine` defaults to `device="cuda"`, so it raises on a CPU-only
host unless you pass `device` explicitly.

## How the loop works

Decider never sees the image. It reads a text rendering of what is known:

```
Task: answer a question about a road scene photograph using vision tools.
Question: What vehicles appear and what text can be read on the sign?
Image: provided and readable.
Evidence collected so far:
  - detect_objects ran successfully. Objects detected: Road (1.00), Sign (1.00),
    Symbol (1.00), Bus Stop (0.96). (tool confidence: 1.00)
Tools not yet run: read_sign_text.
Steps used: 1 of 4.
```

That constraint is the point, not a limitation. A decision over an evidence summary is
portable: the same selector works whether the evidence came from OCR, a SQL query, or
a colleague. The repo ships `examples/triage_example.py`, which drives the identical
engine over support tickets with no image anywhere in it.

Here is a real run, two tools in sequence:

```
step 1: detect_objects   confidence=0.774   P(enough)=0.196   177 ms
        detect_objects 0.832 | read_sign_text 0.158 | request_better 0.006 | answer 0.004
step 2: read_sign_text   confidence=0.984   P(enough)=0.274   160 ms
        read_sign_text 0.989 | request_better 0.004 | answer 0.007
step 3: answer           confidence=0.938   P(enough)=0.582   162 ms
```

Read step 1 carefully. Confidence 0.774, with 0.832 on the detector and 0.158 on OCR —
the model is saying both openings are defensible, because they are. By step 2, after
seeing what the detector found, OCR is at 0.989. **The decision changed because the
evidence changed.** And `P(evidence sufficient)` climbs 0.196 → 0.274 → 0.582 as the
state fills in.

---

## The reusable bit: stop asking the model to respect rules you can enforce

This is the part worth stealing for your own agent.

Preconditions — "this tool already ran", "this needs a file that is missing" — are
checked by a function, not requested in a prompt. The obvious implementation is to
offer all four actions and veto an illegal pick. I built that first. Then I measured
the alternative: remove illegal actions from the menu before the model ever sees them.

Same 37 cases, same model, only the menu changes:

| | all 4 offered | only legal offered |
|---|---|---|
| **Policy overrides** | **40** | **0** |
| Behaviour correct | 97.3 % | 97.3 % |
| Selector latency median | 10 834 ms | 9 119 ms |

Forty of Decider's first-choice picks were illegal in the state they were made in,
mostly re-picking a tool whose output it already had. The outcomes barely move, because
the fallback walks the model's own probability ranking for the best legal alternative.
But the model spent probability mass on forty impossible options, and it got slower
doing it.

The general form: **do not ask a model to respect a constraint you can enforce by not
offering the option.** In the pattern, `Episode.overrides` going up is a diagnostic —
it means your state rendering has stopped carrying a precondition.

The engine is domain-free (`src/roadsense/pattern/evidence_seeking.py`):

```python
seeker = EvidenceSeeker(
    actions=ACTIONS,
    render_state=render,     # (state, progress) -> str
    is_legal=is_legal,       # (state, action)   -> str | None
    execute=execute,         # (state, action)   -> None
    max_steps=4,
    fallback_action="answer",
    mask_illegal=True,
)
episode = seeker.run(my_state, decider)
```

## Latency: a telemetry question, and a hardware class

I measured ~3 200 ms per decision and assumed I had misconfigured something, because
AWS publishes ~115 ms. Two things turned out to be true, and the first one is the
embarrassing one.

**Half the latency was a question nothing read.** RoadSense asked two questions per
step, and the second — `evidence_sufficient` — is pure telemetry. It is recorded,
displayed, and no branch of the control flow touches it. Dropping it:

| | with | without |
|---|---|---|
| Selector latency median | 9 119 ms | **4 229 ms** (−53.6 %) |
| Tool set exactly right | 75.7 % | 75.7 % |
| Behaviour correct | 97.3 % | 97.3 % |
| Answer fidelity | 100 % | 100 % |

Every quality metric identical to the digit. The second question is *cheaper* than
re-encoding the state — the KV cache does its job, and the marginal cost doesn't grow
with state size — but on CPU it is not free, and "barely costs more" is not the same
as "costs nothing".

**The rest was the hardware class.** `causal-conv1d` is a source-only CUDA extension
and `flash-linear-attention` is Triton; neither has a CPU path, so the fallback
warnings `transformers` prints on a CPU host are accurate and unfixable at once. And
the package's own source says what CPU costs — from `infer.py`:

> "on an M3 Pro a 256-token v19 question takes 7.7 s in bf16 and **3.5 s in fp32**"

My 3.2 s was in line with the authors' own PyTorch-CPU number. The 115 ms headline is
a GPU, and the 153 ms MacBook figure is almost certainly the MLX backend.

On an A10G (`g5.xlarge`), same code:

| | CPU | GPU | GPU + `flash-linear-attention` |
|---|---|---|---|
| per decision, 1 question | 1 278 ms | 98 ms | **78 ms** |
| per decision, 2 questions | 2 241 ms | 185 ms | **163 ms** |

`flash-linear-attention` is worth 20–31 %, and its real effect is on scaling: with it,
latency barely moves between a 73-character state and a 1 267-character one.
`causal-conv1d` never finished compiling, so there is headroom I am not claiming.

## The measurements, including where Decider loses

Four selectors, same loop, same tools, same writer. Only the thing choosing changes.
37 cases: 19 answerable, 18 where the right move is to say the evidence is not enough.

| | keyword rule | run everything | LLM selector | **Decider 2B** |
|---|---|---|---|---|
| Tool set exactly right | **100 %** | 45.9 % | 89.2 % | 75.7 % |
| Behaviour correct | 89.2 % | 89.2 % | **100 %** | 97.3 % |
| Invalid inputs handled | 2/5 | 2/5 | **5/5** | **5/5** |
| Per decision | ~0 ms | ~0 ms | 822 ms | **94 ms** |
| Billable selector tokens | 0 | 0 | 23 311 | **0** |

**A ten-line if-statement beats a 2-billion-parameter model on the headline metric.**
Part of that is the metric: `expected_tools` is derived from the question's wording,
which is exactly what a keyword rule reads, so the baseline starts ahead. Part is real
— Decider ran a second tool on nine cases that only needed one, always the same way:
after one tool succeeds, it reaches for the other instead of stopping.

What the keyword rule cannot do is notice the file is missing. On the three cases with
a missing file, a zero-byte file, and no image at all, it answers anyway. Decider reads
a state that says the image is unusable and asks for better evidence. That is the
difference between a router and a decision over evidence.

Against the LLM selector doing the identical job in the identical loop, Decider is
**8.2× faster and consumes no billable tokens at all** — the hybrid argument for small
decision models, measured instead of asserted. The LLM still picks tools more
accurately. Speed and cost are where Decider wins here; accuracy is not.

## What didn't work

**`rs-029`.** A sign degraded until OCR returns the single letter `I` at 0.854
confidence. Expected behaviour was *insufficient*; Decider answered. Hard to blame it —
the state said OCR succeeded with 0.85. Rekognition was confidently wrong rather than
appropriately uncertain.

That generalises. Sweeping degradation levels, Rekognition's OCR confidence on
high-contrast block text is **bimodal**: it reads the word at ~0.98, reads something
else at ~0.85, or returns nothing. No threshold separates right from wrong, because
wrong readings are not low-confidence. Any agent treating tool confidence as a proxy
for tool correctness inherits that blind spot. I left the case in and left the
threshold alone.

**A bug the benchmark caught in my own code.** The fidelity check flagged an answer
quoting text no tool produced. It was real: `read_sign_text` built its summary from
unfiltered `LINE` detections while its structured items came from confidence-filtered
`WORD` detections. A line can clear the floor when its word does not. The summary said
`'STOP | AHEAD | the'`, and Bedrock faithfully repeated a word that was in no item.
Fixed by rebuilding the reading from kept words grouped by `ParentId`; two regression
tests pin it.

**Bedrock went down mid-session.** Haiku 4.5 returned `ServiceUnavailableException`
from two machines at once while other models answered normally. The client now falls
back on capacity errors only — a `ValidationException` still fails loudly — and the
trace records which model actually answered.

## Try it

```bash
pip install -e ".[dev,decider]"
python scripts/smoke_decider.py                 # 3 real decisions
python -m roadsense.cli -q "What vehicles appear and what text can be read on the sign?" \
                        -i demo/images/synthetic_sign_clear.jpg
python examples/triage_example.py               # the pattern, no image in it
python -m roadsense.web.server --port 8080      # watch it decide in a browser
python eval/run_benchmark.py --arms keyword exhaustive llm decider
```

61 tests, Apache-2.0, and every number in this article was generated by a run rather
than typed in — `eval/results/` holds the raw output.

RoadSense reads static image files. It is not a safety system: it does not touch a
vehicle, a camera, CAN/ECU/ADAS, and it never advises on braking or collision risk.
The demonstrated value is the decision pattern, the traceability, and the measurements.

# RoadSense: an agent that decides what evidence to gather, then decides again when it sees it

**Tags:** `#strands-decider-2b` `#tool-selection`
**Repo:** https://github.com/Rodrigo2326/roadsense-agent · **Demo:** https://rodrigo2326.github.io/roadsense-agent/ · Apache-2.0

Ask an agent a question about a photograph and it usually does one of two things: runs
every tool it has, or picks one from the wording of the question and commits. Neither is
a decision. The first is a shopping list; the second is a regex with extra steps.

**RoadSense does neither.** It has four actions, it picks one, it looks at what came
back, and then it picks again — and the second pick is frequently not the one it would
have made first.

**The decision I handed to Strands Decider 2B is exactly that one: which action to take
next, given everything learned so far.** Not how to phrase an answer, not what the
photograph contains — only *what to do next*. Decider scores the four options and
returns a distribution with a calibrated confidence, in about 94 ms, without writing a
single word. Amazon Rekognition gathers the evidence. Amazon Bedrock writes the final
answer and never sees the image, so it cannot describe anything the tools did not find.

Then I measured what handing that decision to a 2B model costs, against Claude doing
the identical job. 37 cases, seven selectors, the same loop, the same tools and the
same writer — only the thing choosing the next action changes:

| | keyword rule | Haiku 4.5 | Sonnet 4.6 | Opus 4.5 | **Decider 2B** |
|---|---|---|---|---|---|
| Tool set exactly right | **100 %** | 89.2 % | 89.2 % | 89.2 % | 75.7 % |
| Invalid inputs handled | 2/5 | **5/5** | **5/5** | **5/5** | **5/5** |
| Selector latency, median | ~0 ms | 1 883 ms | 1 747 ms | 3 453 ms | **230 ms** |
| Billable tokens | 0 | 25 622 | 24 377 | 24 288 | **0** |

Three things there surprised me, and all three are more useful than the agent I built
to produce them.

**Scale buys nothing.** Haiku 4.5, Sonnet 4.6 and Opus 4.5 land on 89.2 % — the same
number to the decimal, across roughly an order of magnitude in price, with Opus taking
twice Haiku's latency to get there. The cheapest frontier model available is also the
joint best, so the gap to Decider is not a capability gap anyone can spend their way
out of.

**A ten-line if-statement beats all of them** on tool accuracy — and is blind to the
thing that matters. It cannot notice the file is missing; on the invalid inputs it
answers anyway, 2 out of 5.

**Decider gives up 13.5 points and runs 7.6× faster than the best of them, for no
billable tokens at all** — 94 ms a decision on an A10G. That is the trade, measured
rather than asserted.

---

## How the agent works

Four actions, taken one at a time, each chosen after seeing what the last one returned:

| action | what it does |
|---|---|
| `detect_objects` | Amazon Rekognition `DetectLabels` |
| `read_sign_text` | Amazon Rekognition `DetectText` |
| `request_better_evidence` | stop, and say what is missing |
| `answer` | stop, and answer from what was gathered |

A real run, on a photograph of a highway:

```
step 1: detect_objects   confidence=0.774   160 ms
        detect_objects 0.832 | read_sign_text 0.158 | request_better 0.006 | answer 0.004
step 2: read_sign_text   confidence=0.978   160 ms
        -> OCR read the text: 'SPEED | LIMIT | 55 | EAST | 6'
step 3: answer           confidence=0.946   161 ms
```

Read step 1 carefully. Confidence 0.774, with 0.83 on the detector and 0.16 on OCR: the
model is saying both openings are defensible, because they are. By step 2, after seeing
what the detector found, OCR is at 0.978. **The decision changed because the evidence
changed.**

Amazon Bedrock writes the final answer and **never sees the image** — only the text the
tools reported. It cannot describe what was not found. That is a property of the
plumbing, not a promise in a prompt.

## Why a pointer model is the right shape here

Decider 2B does not generate text. It scores a fixed list of options and returns a
distribution with a calibrated confidence. That removes a class of failure rather than
mitigating it: the model physically cannot return a name that is not on the menu.

Not theoretical. In the sweep above, **Sonnet 4.5 returned seven replies the parser
could not read** — and the parser is deliberately generous, looking for the documented
JSON and then falling back to scanning the text for any action name. Seven defeated
both paths, and they are the entire difference between its 81.1 % and the 89.2 %
ceiling. Sonnet 4.6 produced none, so this is not a statement about model quality. It
is a reminder that format compliance is something a generative selector can lose and
regain between versions, and that a pointer model never has to have.

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
resp.answers["next_action"].confidence      # (N*p_max - 1)/(N - 1), calibrated
```

Two notes for anyone wiring this up. `ChoiceQuestion` takes **`criteria`**, not
`options` — I read that off `strands_decider/schema.py` after published write-ups
disagreed. And `load_engine` defaults to `device="cuda"`, so it raises on a CPU-only
host unless you pass `device` explicitly.

---

## The reusable bit: stop asking a model to respect rules you can enforce

This is the part worth taking to your own agent.

Preconditions — "this tool already ran", "this needs a file that is missing" — are a
function, not a prompt instruction. The obvious implementation offers all four actions
and vetoes an illegal pick. I built that first, then measured the alternative: remove
illegal actions from the menu before the model ever sees them.

Same 37 cases, same model, only the menu changes:

| | all four offered | only legal offered |
|---|---|---|
| **Policy overrides** | **40** | **0** |
| Behaviour correct | 97.3 % | 97.3 % |
| Selector latency median | 10 834 ms | 9 119 ms |

Forty of Decider's first-choice picks were illegal in the state they were made in,
mostly re-picking a tool whose output it already had. The outcomes barely move, because
the fallback walks the model's own ranking for the best legal alternative. But the model
spent probability mass on forty impossible options, and got slower doing it.

**Do not ask a model to respect a constraint you can enforce by not offering the
option.** In the pattern, a rising override count is a diagnostic: it means your state
rendering has stopped carrying a precondition.

The engine is domain-free. `src/roadsense/pattern/evidence_seeking.py` has no road, no
image and no AWS call in it:

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

`examples/triage_example.py` drives the identical engine over support tickets, where the
refund lookup is not offered on step one because it depends on a delivery date nobody
has looked up yet. Same loop, no image anywhere.

---

## Latency: a telemetry question, and a hardware class

I measured ~3 200 ms a decision and assumed a misconfiguration, because AWS publishes
~115 ms. Two things were true, and the first is embarrassing.

**Half the latency was a question nothing read.** The agent asked two questions per
step, and the second — `evidence_sufficient` — is pure telemetry: recorded, displayed,
touched by no branch of the control flow. Dropping it cut selector latency by 53.6 %
with every quality metric identical to the digit.

**The rest was the hardware class.** `causal-conv1d` is a source-only CUDA extension and
`flash-linear-attention` is Triton, so on a CPU host the fallback warnings
`transformers` prints are accurate *and* unfixable at the same time. The package's own
source says what CPU costs — from `infer.py`: *"on an M3 Pro a 256-token v19 question
takes 7.7 s in bf16 and 3.5 s in fp32"*. My 3.2 s was in line with the authors' own
number. The 115 ms headline is a GPU.

On an A10G, same code: **78 ms** for one question, 163 ms for two.
`flash-linear-attention` installs cleanly there and is worth 20–31 %; its real effect is
on scaling, since latency barely moves between a 73-character state and a
1 267-character one. `causal-conv1d` never finished compiling, so there is headroom I am
not claiming.

---

## What didn't work

**`rs-029`.** A sign degraded until OCR returns the single letter `I` at 0.854
confidence. Expected behaviour was *insufficient*; Decider put 0.92 on answering. Hard
to blame it: the state said OCR succeeded at 0.85. Rekognition was confidently wrong
rather than appropriately uncertain.

That generalises. Sweeping degradation levels, Rekognition's OCR confidence on
high-contrast block text is **bimodal** — it reads the word at ~0.98, reads something
else at ~0.85, or returns nothing. No threshold separates right from wrong, because
wrong readings are not low-confidence. Any agent treating tool confidence as a proxy for
tool correctness inherits that blind spot. The case is still in the benchmark, and on
the demo page, scored as a loss.

**A bug the benchmark caught in my own code.** The fidelity check flagged an answer
quoting text no tool had produced. It was real: `read_sign_text` built its summary from
unfiltered `LINE` detections while its items came from confidence-filtered `WORD`
detections, and a line can clear the floor when its word does not. The summary said
`'STOP | AHEAD | the'`, and Bedrock faithfully repeated a word that was in no item.
Fixed, with two regression tests.

**Real photographs carry real number plates.** The demo uses four Commons photographs
(three CC0, one CC BY). On the street scene, Rekognition read `MP 7866` and `GF 4205`
straight off it. The fetch script now blurs every text region it can find and re-reads
the file afterwards, failing if any word survives — and the attribution file says what
was changed rather than claiming the photograph was clean.

**Bedrock went down mid-project.** Haiku 4.5 returned `ServiceUnavailableException` from
two machines at once while other models answered normally. The client now falls back on
capacity errors only — a `ValidationException` still fails loudly — and the trace records
which model actually answered.

---

## Try it

The [demo page](https://rodrigo2326.github.io/roadsense-agent/) replays nine real
episodes step by step, four of them on photographs, including the one it gets wrong. It
is static: no backend, no keys, nothing to spend. The live server that runs *your*
photograph is deliberately not hosted, because every click is two Rekognition calls, a
Bedrock completion and GPU time — it ships with run caps, a samples-only mode and a
shared token for anyone who wants to expose it anyway.

```bash
pip install -e ".[dev,decider]"
python scripts/smoke_decider.py                 # 3 real decisions
python -m roadsense.cli -q "What vehicles appear and what text can be read on the sign?" \
                        -i demo/images/synthetic_sign_clear.jpg
python examples/triage_example.py               # the pattern, no image in it
python -m roadsense.web.server --port 8080      # watch it decide in a browser
python eval/run_benchmark.py --arms keyword exhaustive llm decider
```

78 tests, Apache-2.0, and every number above was generated by a run rather than typed
in. `eval/results/` holds the raw output, and the demo page is rebuilt from it on every
push, so the demo cannot drift away from the benchmark it claims to show.

RoadSense reads static image files. It is not a safety system: it does not touch a
vehicle, a camera, CAN/ECU/ADAS, and it never advises on braking or collision risk. The
demonstrated value is the decision pattern, the traceability, and the measurements.

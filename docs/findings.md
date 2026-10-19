# What the benchmark actually showed

37 cases, four selectors, one run: `eval/results/report_full.md`. Plus a second run
of the Decider arm with illegal-action masking turned off: `report_nomask.md`.
Everything below is from those runs.

The short version: **Decider 2B is not the most accurate tool selector in this
benchmark, it is by far the cheapest and fastest one that still notices when there is
nothing to work with.** Both halves of that sentence matter.

---

## 1. Decider loses on tool-set accuracy. The metric is part of why.

| | keyword | exhaustive | llm | decider |
|---|---|---|---|---|
| Tool set exactly right | **100%** | 45.9% | 89.2% | 75.7% |
| Unnecessary calls (total) | **0** | 20 | 4 | 9 |

A ten-line if-statement beats a 2-billion-parameter model on the headline metric.

Part of that is a property of the metric, and it would be dishonest not to say so:
`expected_tools` is derived from the **question wording** — "what does the sign say"
needs OCR, "what objects are visible" needs the detector. That is precisely what the
keyword rule reads. The ground truth and the baseline are looking at the same signal,
so the baseline starts the race ahead.

Part of it is real. Decider's 9 unnecessary calls follow one pattern:

```
rs-001, rs-003, rs-006, rs-007, rs-009   expected [detect_objects]   ran both
rs-017, rs-018, rs-019, rs-020           expected [read_sign_text]   ran both
```

After one tool succeeds, Decider reaches for the other instead of answering. It
undervalues "stop now" whenever more evidence is still available. For a question that
only needed one tool, that is a wasted Rekognition call and ~3 s of latency.

## 2. Where Decider earns its place: knowing there is nothing to work with

| | keyword | exhaustive | llm | decider |
|---|---|---|---|---|
| Invalid-input cases right | 2/5 | 2/5 | **5/5** | **5/5** |
| Correct on insufficient | 77.8% | 77.8% | **100%** | 94.4% |
| Behaviour correct overall | 89.2% | 89.2% | **100%** | 97.3% |

Three cases separate them — `rs-034`, `rs-036`, `rs-037`: a missing file, a zero-byte
file, and no image argument at all.

The keyword rule answers anyway. It has no concept of state; it reads the question,
matches a word, and commits. Decider reads a state that says the image is unusable
and asks for better evidence. That is the difference between a router and a decision
over evidence, and it is the thing this project is actually about.

## 3. Masking illegal actions: 40 overrides removed, same answers

Same 37 cases, Decider arm, only the menu changes:

| | masked (default) | unmasked |
|---|---|---|
| Policy overrides | **0** | **40** |
| Tool set exactly right | 75.7% | 73.0% |
| Behaviour correct | 97.3% | 97.3% |
| Selector latency median | 9,119 ms | 10,834 ms |

Forty of Decider's first-choice picks were illegal in the state they were made in —
mostly re-picking a tool whose output it already had. The final outcomes barely move,
because the fallback walks the model's own ranking and finds a legal alternative. But
the model spent probability mass on forty impossible options, and the loop spent
~1.7 s per case more getting there.

This is the pattern's load-bearing claim, with a number on it: **do not ask a model
to respect a constraint you can enforce by not offering the option.** `Episode.overrides`
going up is a signal that your state rendering has stopped carrying a precondition.

## 4. The case Decider got wrong, and why it is interesting

`rs-029` — the sign degraded to 65 px. OCR returns the single letter `I` at **0.854**
confidence. The expected behaviour was *insufficient*; Decider answered.

It is hard to blame the model. The state it read said OCR succeeded with 0.85
confidence. Nothing in the evidence suggested a problem, because Rekognition was
confidently wrong rather than appropriately uncertain.

This generalises past this benchmark. A sweep over degradation levels found that
Rekognition's OCR confidence on high-contrast block text is **bimodal**: it reads the
word at ~0.98, reads something else at ~0.85, or returns nothing. There is no
threshold that separates right from wrong, because wrong readings are not low
confidence. Any agent — this one included — that treats tool confidence as a proxy
for tool correctness inherits that blind spot.

We left the case in, and left the threshold alone.

## 5. Cost and latency: the honest trade

| | keyword | llm | decider |
|---|---|---|---|
| Selector latency median | ~0 ms | 1,559 ms | **9,119 ms** |
| Wall clock median | 1,530 ms | 3,181 ms | **10,997 ms** |
| Selector tokens (billable) | 0 | 23,311 in / 2,242 out | **0** |
| Rekognition USD (verified rate) | $0.044 | $0.048 | $0.053 |

Those are the CPU numbers, and they are misleading on their own. Section 5b takes
them apart: half the cost was a telemetry question nothing read, and the rest was a
hardware class rather than a misconfiguration. On a GPU with the same code, Decider
runs at **94 ms per decision and is 8.2x faster than the LLM selector**. The table
above is kept as the CPU baseline because reproducing it needs no GPU quota.

Its selector tokens are not billable, because it runs on an EC2 instance charged by
the hour rather than per call. Those are two different cost shapes and the report
keeps them in separate rows rather than adding them into one misleading total. Which
is cheaper depends entirely on volume: at 37 cases the hourly instance loses badly;
the crossover is a function of request rate, and this benchmark does not measure it.

## 5b. Latency: a telemetry question, and a hardware class

The published headline for Decider 2B is **~115 ms** on an RTX 3090 and **~153 ms**
on an M3 MacBook. RoadSense measured **~3 200 ms** per decision. A 20x gap is usually
a misconfiguration, so we went looking for one.

### The kernels cannot be the answer *on CPU*

`transformers` warns at every load that `causal_conv1d` and `flash-linear-attention`
are missing and that it is falling back to reference PyTorch. That looks like the
culprit, and on a GPU it partly is. On a CPU host it cannot be: `causal-conv1d` ships
**source-only and builds against CUDA**, and `flash-linear-attention` is Triton.
Neither has a CPU path, so on the CPU box there was nothing to install. The warning
is accurate and the fix is unavailable at the same time.

### The package already told us what CPU costs

From `strands_decider/infer.py`, in the docstring of `_upcast_torso_for_cpu`:

> "on an M3 Pro a 256-token v19 question takes 7.7 s in bf16 and **3.5 s in fp32**"

That is the package authors' own PyTorch-CPU measurement. Our 3.2 s on a c7i.2xlarge
is in line with it, slightly better. **The 115 ms figure is a GPU number, and the
153 ms figure is almost certainly the MLX backend the repo ships for Apple Silicon —
neither is PyTorch on CPU.** Our latency was never a misconfiguration; it was a
hardware class.

### What we measured instead

`scripts/profile_latency.py`, on the c7i.2xlarge (8 vCPU, fp32 torso):

| state | 1 question | 2 questions | cost of the 2nd |
|---|---|---|---|
| 73 chars | 1 044 ms | 2 003 ms | +959 ms |
| 303 chars | 1 278 ms | 2 241 ms | +963 ms |
| 1 267 chars | 3 243 ms | 4 044 ms | +801 ms |

Two results worth stating plainly, because both contradict a reasonable prior:

**Thread count does nothing.** torch defaulted to 4 of 8 vCPUs. Forcing 8 made it
marginally *slower* at every state size. That lever does not exist here.

**A second question is not free.** The model's headline property is that the state is
encoded once and its KV cache broadcast across the question batch, so extra questions
"barely cost more time". The cache does work — the marginal cost does not grow with
state size, which is exactly what a shared prefix predicts. But on CPU the per-question
forward pass dominates, and that costs ~0.8–1.0 s flat. Cheap relative to re-encoding
the state; not cheap in absolute terms.

### Finding 1: half the latency was a number we only logged

RoadSense asked two questions per step. The second, `evidence_sufficient`, is
**telemetry**: recorded, displayed, read by nothing. No branch in the control flow
touches it.

Same 37 cases, decider arm, the only change being whether that question is asked:

| | with sufficiency | without |
|---|---|---|
| Selector latency median (CPU) | 9 119 ms | **4 229 ms** (-53.6 %) |
| Tool set exactly right | 75.7 % | 75.7 % |
| Behaviour correct | 97.3 % | 97.3 % |
| Correct on insufficient | 94.4 % | 94.4 % |
| Answer fidelity | 100 % | 100 % |

Every quality metric identical to the digit. **On a decision model, a question you
only log is not free — know what it costs before you keep it.**

### Finding 2: the hardware class was the whole story

With a GPU quota approved, the same code ran on a `g5.xlarge` (A10G 24 GB). The
torso stays in bfloat16 on CUDA instead of being upcast to fp32 as it is on CPU.

`scripts/profile_latency.py`, per decision:

| state | CPU (c7i, fp32) | GPU, no kernels | GPU + `flash-linear-attention` |
|---|---|---|---|
| 73 chars, 1 question | 1 044 ms | 92 ms | **77 ms** |
| 303 chars, 1 question | 1 278 ms | 98 ms | **78 ms** |
| 1 267 chars, 1 question | 3 243 ms | 125 ms | **86 ms** |
| 1 267 chars, 2 questions | 4 044 ms | 212 ms | **163 ms** |

`flash-linear-attention` installs cleanly on CUDA and is worth 20-31 %. Its real
effect is on *scaling*: with it, latency barely moves between a 73-char state and a
1 267-char one (77 → 86 ms), which is what an optimised linear-attention kernel is
supposed to do. `causal-conv1d` never finished compiling — it is a source-only CUDA
extension and nvcc was still going after 20 minutes on 4 vCPUs — so the numbers above
still carry its reference-PyTorch fallback. There is headroom left.

### The two findings together

Full benchmark, 37 cases, decider arm, every configuration:

| | CPU 2q | CPU 1q | GPU 2q | GPU 1q | LLM selector |
|---|---|---|---|---|---|
| Selector latency median | 9 119 ms | 4 229 ms | 479 ms | **230 ms** | 1 890 ms |
| Per decision | 3 752 ms | 1 740 ms | 197 ms | **94 ms** | 822 ms |
| Wall clock median | 10 997 ms | 5 824 ms | 2 227 ms | **1 964 ms** | 3 522 ms |
| Tool set exactly right | 75.7 % | 75.7 % | 75.7 % | 75.7 % | 89.2 % |
| Behaviour correct | 97.3 % | 97.3 % | 97.3 % | 97.3 % | 100 % |
| Billable selector tokens | 0 | 0 | 0 | **0** | 23 311 |

**94 ms per decision beats the 115 ms AWS publishes for an RTX 3090**, and every
quality metric is unchanged across all four Decider configurations — the 40x speedup
from the worst to the best configuration cost nothing in accuracy.

Against the LLM selector doing the identical job in the identical loop, Decider on a
GPU is **8.2x faster and consumes no billable tokens at all**. That is the hybrid
argument AWS makes for these models, measured rather than asserted: hand the rote
decision to the small model, keep the generative model for the writing.

The honest caveat stays attached: the LLM arm still picks tools better (89.2 % vs
75.7 %) and handles the insufficient cases perfectly (100 % vs 94.4 %). Speed and
cost are where Decider wins; accuracy on this benchmark is not.

## 5c. Would a better model just win? We bought three and asked.

Comparing a 2B pointer model against only the cheapest frontier option would be
rigging the baseline, so the `llm` arm was re-run across a price tier: Claude
Haiku 4.5, Sonnet 4.5 and Opus 4.5, same 37 cases, same loop, same prompt. (Opus 5,
Fable 5 and Sonnet 5 return `AccessDeniedException` on this account and could not be
included.)

| | keyword | haiku 4.5 | sonnet 4.5 | opus 4.5 | **Decider 2B** |
|---|---|---|---|---|---|
| Tool set exactly right | **100 %** | 89.2 % | 81.1 % | 89.2 % | 75.7 % |
| Behaviour correct | 89.2 % | **100 %** | 97.3 % | 97.3 % | 97.3 % |
| Unnecessary calls | **0** | 4 | 7 | 4 | 9 |
| Unparseable replies | – | 0 | **7** | 0 | – |
| Selector latency median | ~0 ms | 1 883 ms | 4 280 ms | 3 453 ms | **230 ms** |
| Billable selector tokens | 0 | 23 353 | 24 244 | 23 353 | **0** |

**Scale buys nothing here.** Opus 4.5 ties Haiku 4.5 to the decimal — 89.2 % both —
and costs twice the latency to get there. Sonnet 4.5 lands *below* both. Whatever
separates 89.2 % from Decider's 75.7 % is not a capability gap you can spend your way
out of: three tiers of frontier model plateau at the same number.

That reframes the headline loss. Decider gives up 13.5 points of tool-set accuracy
against the best available selector — and in exchange runs **15× faster than Opus**
with **zero billable tokens**. Those are the real terms of the trade, and they are
better terms than the single-baseline comparison suggested.

### The seven replies nobody could read

Sonnet 4.5's 81.1 % has a specific cause: **seven of its selector replies could not be
parsed.** Haiku and Opus produced none.

The parser is deliberately generous — it looks for the documented JSON, and failing
that scans the text for any action name, because the point is to compare selection
quality rather than to score models on JSON discipline. Seven replies defeated both.

This is the architectural argument made concrete. A generative model asked to pick
from a menu can drift off format, and when it does the agent has to guess what it
meant. A pointer model cannot: it scores a fixed list and returns a distribution.
"The selector emitted something unusable" is not a failure mode that exists for
Decider 2B — not because the handling is better, but because there is no slot for it.
It cost Sonnet eight points of accuracy on a task where the model itself is perfectly
capable.

## 6. What we would do next

- **Fix the over-calling directly.** The state already says which tools have run; it
  does not say which ones the *question* needs. Adding that to the rendering is the
  obvious next experiment, and it is measurable with the harness as it stands.
- **Try the larger Decider checkpoints.** Only `2B-qwen3.5-v1-2610` was tested. Six
  others exist, including gemma4 variants up to 26B. The frontier-LLM sweep showed
  scale does not help a *generative* selector here; whether it helps a *pointer* model
  is a different question and an open one.
- **Finish the kernel story.** `causal-conv1d` never finished building. Its fallback
  warning is still printed on every GPU load, so there is measurable headroom below
  77 ms that we have not claimed.
- **Replace synthetic images with photographs.** The degradation dial is reproducible
  but it is not what a real camera does, and the drawn vehicles are not recognised as
  vehicles at all.

## Scoreboard, plainly

| Arm | Wins | Loses |
|---|---|---|
| `keyword` | tool accuracy, latency, cost | blind to state; answers with no image |
| `exhaustive` | never misses evidence | 20 unnecessary calls; worst tool accuracy |
| `llm` | best behaviour overall; 5/5 on invalid input; best tool accuracy of any model | 23k billable selector tokens; 7 unparseable replies at the Sonnet tier; scaling to Opus buys nothing |
| `decider` | fastest selector on GPU (94 ms/decision, 8.2x the LLM arm); 5/5 on invalid input; zero billable tokens | 9 unnecessary calls; needs a GPU to be fast |

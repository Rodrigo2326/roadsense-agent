# Evidence-Seeking Tool Selection

**A reusable Strands Decider pattern.** Tags: `#strands-decider-2b` `#tool-selection`

Implementation: [`src/roadsense/pattern/evidence_seeking.py`](../src/roadsense/pattern/evidence_seeking.py)
Domain-free example: [`examples/triage_example.py`](../examples/triage_example.py)
Tests: [`tests/test_pattern.py`](../tests/test_pattern.py)

---

## The problem

You have a question, and several actions that each buy you a different piece of
evidence. You do not know up front which ones you need.

Running them all is wasteful, and often impossible — some actions have preconditions
that only become true after an earlier action has run. Answering from the first one
is how agents end up confidently wrong. And a text-generating model asked to "pick a
tool" can emit a tool that does not exist, emit malformed arguments, or quietly
invent the evidence it was supposed to go and fetch.

## The shape of the answer

Loop: render the state as text → ask a decision model which action to take next →
enforce legality in code → execute → fold the result back into the state. Stop when
the model picks a terminal action, or when the budget runs out.

```
             ┌─────────────────────────────────────────┐
             │  state (text: what is known so far)     │
             └────────────────────┬────────────────────┘
                                  ▼
                       ┌──────────────────────┐
     menu of LEGAL ───▶│  Decider 2B (choice) │──▶ probabilities + confidence
     actions only      └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ legality re-check    │──▶ override recorded, with reason
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ execute → new facts  │──┐
                       └──────────────────────┘  │
                                  ▲              │
                                  └──────────────┘   until terminal or budget
```

## The four moving parts

### 1. A typed, closed action set

Every option is named and described up front. Decider 2B is a *pointer* model: it
scores a fixed list of options rather than generating a name. "The model emitted an
action that does not exist" is not a failure mode that can occur — not because we
validate it away, but because the architecture has no slot for it.

```python
ACTIONS = [
    Action("lookup_order", "Look up the order to find its status and ship date."),
    Action("check_refund_window", "Check whether the order is still inside the refund window."),
    Action("ask_customer", "Stop and ask the customer for a fact no system holds.", terminal=True),
    Action("answer", "Stop looking things up and reply from the facts gathered.", terminal=True),
]
```

The `description` is not documentation. It is the text the pointer head reads, and it
is the main lever you have on decision quality. Write it as a plain statement of what
the action does and when it applies.

### 2. A state rendered as text

The decision model sees what has been learned, never the raw artefact. RoadSense
never shows Decider the image; the triage example never shows it the database.

This is what makes the selector portable. A decision over an evidence summary works
the same whether the evidence came from OCR, a SQL query, or a colleague — and it is
what lets you unit-test the decision layer with a string.

Keep in the rendered state: the question, what each tool returned **including the
tool's own confidence**, what has not been run yet, what failed, and how much budget
is left. Keep out: anything you would not want the model to repeat as fact.

### 3. Legality enforced in code

Preconditions are a function, not a prompt instruction:

```python
def is_legal(state, action) -> str | None:
    if action in state.ran:
        return f"{action} already ran; repeating it would add no new facts"
    if action == "lookup_order" and not state.order_id:
        return "lookup_order needs an order id and the ticket does not have one"
    return None
```

By default illegal actions are **masked out of the menu** rather than offered and
then vetoed. This is not cosmetic — see the measurement below.

When a pick is overruled anyway (the unmasked arm, or a remote backend returning
something odd), the fallback walks the model's *own* probability ranking for the best
legal alternative. The model still chooses; legality only removes what it was never
allowed to pick. The override is recorded with its reason, never silently rewritten.

### 4. A budget, and an honest end

The loop always terminates. When the budget runs out the step is recorded with
`decider_called=False` and `confidence=None` — because nothing decided it, and
writing `0.0` there would read like a decision that happened.

---

## The measurement that justifies masking

RoadSense ran both arms on the same cases with the real Decider 2B checkpoint
(`strands-decider-2B-qwen3.5-v1-2610`, CPU, `c7i.2xlarge`):

| Case | All 4 actions offered | Only legal actions offered |
|---|---|---|
| No image provided | picked `detect_objects` at p=0.907 → vetoed | picked `request_better_evidence`, **confidence 0.962**, 0 overrides |
| OCR already ran and returned nothing | insisted on `read_sign_text` at p=0.891 → vetoed twice | 0 overrides, clean path |

The model does not reliably infer "this tool already ran, so it is pointless" from
the state text, even when the state says so in plain words. Masking does not paper
over that — it removes the question. And the confidence on the action it *does* pick
goes up, because the probability mass stops being spent on options it could never
have taken.

**`Episode.overrides` is therefore a diagnostic, not a log.** A rising override count
means your action descriptions or your state rendering have stopped carrying the
preconditions.

---

## Using it

```python
from roadsense.pattern import Action, EvidenceSeeker

seeker = EvidenceSeeker(
    actions=ACTIONS,
    render_state=render,        # (state, progress) -> str
    is_legal=is_legal,          # (state, action)   -> str | None
    execute=execute,            # (state, action)   -> None, mutates state
    max_steps=4,
    fallback_action="answer",   # must be terminal, and legal in every state
    mask_illegal=True,
)

episode = seeker.run(my_state, decider)
```

`decider` is anything with
`decide(state_text: str, options: dict[str, str]) -> SelectorChoice`.
`roadsense.decider_client.DeciderResult` satisfies that protocol structurally, so the
Strands Decider client — in-process or over `strands-decider serve` — drops in with
no adapter.

### Wiring Decider 2B

The Decider call underneath is one `choice` question. RoadSense pairs it with a
`noul` question in the *same* call, because asking a second question about the same
state is nearly free — the state's KV cache is encoded once and broadcast across the
question batch:

```python
engine.ask(state=state_text, questions={
    "next_action": ChoiceQuestion(instructions="Which single action should the agent "
                                  "take next to answer the question with evidence it "
                                  "can actually verify?", criteria=options),
    "evidence_sufficient": NoulQuestion(instructions="Is the evidence collected so far "
                                        "already sufficient to answer the question "
                                        "without guessing?"),
})
```

Note `criteria`, not `options` — verified against `strands_decider/schema.py` in the
installed package.

---

## When this pattern fits

**Good fit.** Several actions, each gathering different evidence; preconditions
between them; a real cost to running the wrong one; and a case where the right answer
is "I cannot tell from this".

**Bad fit.** One obvious tool. Actions that need generated arguments — Decider picks
*which*, never *with what*; generate arguments separately once the action is chosen.
Decisions that genuinely need to see the raw artefact rather than a summary of it.

## Pitfalls

- **Thin action descriptions.** They are the prompt. One-word descriptions destroy
  discrimination between options.
- **A state that hides preconditions.** If the state does not say a tool already ran,
  neither masking nor the model can help.
- **Treating confidence as accuracy.** Decider's confidence is calibrated
  concentration — `(N·p_max − 1)/(N − 1)` — not a promise that the pick is right. A
  near-split between two reasonable actions is the model being honest.
- **No terminal action that is always legal.** Then the loop can get stuck. The
  constructor rejects this at construction time rather than at 3 a.m.

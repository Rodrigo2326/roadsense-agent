"""Build a self-contained HTML viewer for RoadSense decision traces.

A trace is a JSON document, and JSON does not show what is interesting about it: that
the menu of allowed actions *shrinks* as evidence accumulates, and that the model's
confidence moves with the evidence rather than staying flat. Both are visible at a
glance once the four actions are drawn in a fixed order and the withdrawn ones are
left in place, greyed, instead of silently disappearing.

Reads the benchmark results and emits one standalone page with the traces and images
embedded. No network, no build step.

    python scripts/make_trace_viewer.py
    python scripts/make_trace_viewer.py --cases rs-021 rs-030 rs-037
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Fixed display order, so the shrinking menu reads as a shrinking menu rather than
#: as a reshuffle. Gather-evidence actions first, then the two ways to stop.
ACTION_ORDER = ["detect_objects", "read_sign_text", "request_better_evidence", "answer"]

ACTION_LABEL = {
    "detect_objects": "detect objects",
    "read_sign_text": "read sign text",
    "request_better_evidence": "request better evidence",
    "answer": "answer",
}

CAPTIONS = {
    "rs-021": "A readable sign. Two tools, in sequence — the second chosen only after the first reported back.",
    "rs-030": "The same question, a sign degraded past reading. OCR returns nothing, and the agent says so.",
    "rs-037": "No image at all. Both vision tools are withdrawn before the model ever sees the menu.",
}


def embed_image(path: str | None) -> str | None:
    if not path:
        return None
    name = pathlib.Path(path).name
    local = ROOT / "demo" / "images" / name
    if not local.is_file() or local.stat().st_size == 0:
        return None
    mime = mimetypes.guess_type(local.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(local.read_bytes()).decode()


def collect(results_path: pathlib.Path, case_ids: list[str]) -> list[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    decider = next(r for r in data["results"] if r["arm"] == "decider")
    by_id = {t["case_id"]: t for t in decider["traces"]}

    out = []
    for cid in case_ids:
        t = by_id[cid]
        steps = []
        # Evidence arrives after the step that asked for it; tool steps consume the
        # evidence list in order, so they can be zipped positionally.
        ev_iter = iter(t["evidence"])
        for s in t["steps"]:
            evidence = None
            if s["action"] != "answer":
                evidence = next(ev_iter, None)
            steps.append(
                {
                    "n": s["step"],
                    "action": s["action"],
                    "confidence": s["confidence"],
                    "sufficient": s["evidence_sufficient"],
                    "latency": round(s["latency_ms"]),
                    "offered": s["offered_actions"],
                    "probs": s["probabilities"],
                    "rejected": s["rejected"],
                    "evidence": None
                    if evidence is None
                    else {
                        "tool": evidence["tool"],
                        "ok": evidence["ok"],
                        "quality": evidence["quality"],
                        "summary": evidence["summary"],
                    },
                }
            )
        out.append(
            {
                "id": cid,
                "question": t["question"],
                "image": embed_image(t["image_path"]),
                "caption": CAPTIONS.get(cid, ""),
                "outcome": t["outcome"],
                "answer": t["answer"],
                "steps": steps,
                "model": t.get("decider_backend", ""),
            }
        )
    return out


PAGE = """<title>RoadSense Decision Trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #f1f3f6;
  --surface: #ffffff;
  --sunk: #e7eaf0;
  --ink: #14171c;
  --muted: #5a6472;
  --faint: #8d96a5;
  --rule: #d8dde6;
  --chosen: #2f3fb8;
  --on-chosen: #ffffff;
  --chosen-soft: #dfe3f7;
  --dim: #c3cada;
  --warn: #9c4a09;
  --warn-soft: #f6e8d8;
  --good: #0e6b7d;
  --shadow: 0 1px 2px rgba(20,23,28,.06), 0 8px 24px -16px rgba(20,23,28,.3);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0e1116;
    --surface: #161a21;
    --sunk: #1c212a;
    --ink: #e9ecf1;
    --muted: #949dad;
    --faint: #6d7686;
    --rule: #272d38;
    --chosen: #8c99ff;
    --on-chosen: #0e1116;
    --chosen-soft: #232a4a;
    --dim: #39414f;
    --warn: #e0a165;
    --warn-soft: #2e2418;
    --good: #5ab6c7;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"] {
  --ground: #0e1116;
  --surface: #161a21;
  --sunk: #1c212a;
  --ink: #e9ecf1;
  --muted: #949dad;
  --faint: #6d7686;
  --rule: #272d38;
  --chosen: #8c99ff;
  --on-chosen: #0e1116;
  --chosen-soft: #232a4a;
  --dim: #39414f;
  --warn: #e0a165;
  --warn-soft: #2e2418;
  --good: #5ab6c7;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 48px 24px 96px; }

header.masthead { border-bottom: 1px solid var(--rule); padding-bottom: 28px; margin-bottom: 36px; }
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--faint); margin: 0 0 14px;
}
h1 {
  font-family: "Instrument Serif", Georgia, serif;
  font-weight: 400; font-size: clamp(38px, 6vw, 60px);
  line-height: 1.02; letter-spacing: -.012em;
  margin: 0 0 14px; text-wrap: balance;
}
h1 em { font-style: italic; color: var(--chosen); }
.standfirst { max-width: 62ch; color: var(--muted); font-size: 16.5px; margin: 0; }

.cases { display: flex; flex-wrap: wrap; gap: 8px; margin: 32px 0 28px; }
.case-btn {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; letter-spacing: .02em;
  padding: 9px 15px; border-radius: 999px;
  border: 1px solid var(--rule); background: var(--surface); color: var(--muted);
  cursor: pointer; transition: border-color .15s, color .15s, background .15s;
}
.case-btn:hover { border-color: var(--dim); color: var(--ink); }
.case-btn[aria-selected="true"] {
  background: var(--chosen); border-color: var(--chosen); color: var(--on-chosen);
}
.case-btn:focus-visible { outline: 2px solid var(--chosen); outline-offset: 2px; }

.brief {
  display: grid; grid-template-columns: minmax(0,1fr) 220px; gap: 28px;
  align-items: start; margin-bottom: 40px;
}
@media (max-width: 680px) { .brief { grid-template-columns: 1fr; } }
.question {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 27px; line-height: 1.25; margin: 0 0 12px; text-wrap: balance;
}
.caption { color: var(--muted); font-size: 15px; margin: 0; max-width: 56ch; }
.shot {
  width: 100%; border-radius: 4px; border: 1px solid var(--rule); display: block;
  background: var(--sunk);
}
.no-shot {
  border: 1px dashed var(--rule); border-radius: 4px; padding: 26px 16px;
  text-align: center; color: var(--faint);
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
  letter-spacing: .06em; text-transform: uppercase;
}

.step { border-top: 1px solid var(--rule); padding: 26px 0 4px; }
.step-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
.step-n {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint);
}
.step-pick { font-family: "Instrument Serif", Georgia, serif; font-size: 23px; }
.menu-count {
  margin-left: auto; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; color: var(--faint); font-variant-numeric: tabular-nums;
}

.bars { display: flex; flex-direction: column; gap: 7px; }
.bar-row {
  display: grid; grid-template-columns: 186px minmax(0,1fr) 58px;
  align-items: center; gap: 14px;
}
@media (max-width: 560px) { .bar-row { grid-template-columns: 136px minmax(0,1fr) 50px; } }
.bar-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12.5px; color: var(--muted); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.bar-track { display: block; height: 20px; background: var(--sunk); border-radius: 2px; overflow: hidden; }
.bar-fill {
  /* Must be block: a bare span is inline, and width does not apply to inline
     boxes -- the bars render as empty tracks and the bug is invisible in code. */
  display: block; height: 100%; width: 0; background: var(--dim); border-radius: 2px;
  transition: width .62s cubic-bezier(.22,.8,.3,1);
}
.bar-val {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; color: var(--muted); text-align: right;
  font-variant-numeric: tabular-nums;
}
.is-chosen .bar-fill { background: var(--chosen); }
.is-chosen .bar-label { color: var(--ink); font-weight: 500; }
.is-chosen .bar-val { color: var(--chosen); font-weight: 500; }
.withdrawn .bar-label { color: var(--faint); text-decoration: line-through; text-decoration-thickness: 1px; }
.withdrawn .bar-track { background: transparent; border: 1px dashed var(--rule); }
.withdrawn .bar-val { color: var(--faint); font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; }

.readouts { display: flex; gap: 22px; flex-wrap: wrap; margin: 16px 0 0; }
.readout { display: flex; align-items: baseline; gap: 7px; }
.readout-k {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--faint);
}
.readout-v {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 14px; font-variant-numeric: tabular-nums; color: var(--ink);
}
.readout-v.good { color: var(--good); }

.evidence {
  margin: 18px 0 2px; padding: 14px 16px; background: var(--surface);
  border: 1px solid var(--rule); border-left: 2px solid var(--chosen);
  border-radius: 3px; box-shadow: var(--shadow);
}
.evidence.empty { border-left-color: var(--warn); }
.ev-head {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--faint); margin-bottom: 6px;
  display: flex; gap: 10px; flex-wrap: wrap;
}
.ev-body { font-size: 14.5px; color: var(--ink); margin: 0; }
.note {
  margin-top: 12px; padding: 10px 13px; border-radius: 3px;
  background: var(--warn-soft); color: var(--warn);
  font-size: 13.5px;
}

.outcome { border-top: 1px solid var(--rule); margin-top: 30px; padding-top: 26px; }
.tag {
  display: inline-block; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  padding: 4px 9px; border-radius: 3px; margin-bottom: 14px;
  background: var(--chosen-soft); color: var(--chosen);
}
.tag.insufficient { background: var(--warn-soft); color: var(--warn); }
.answer {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 22px; line-height: 1.42; margin: 0; max-width: 60ch; text-wrap: pretty;
}

footer {
  margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--rule);
  color: var(--faint); font-size: 13px;
}
footer code {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; color: var(--muted);
}
@media (prefers-reduced-motion: reduce) {
  .bar-fill { transition: none; }
}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Strands Decider 2B &middot; tool selection</p>
    <h1>Watch the menu <em>shrink</em></h1>
    <p class="standfirst">
      Every row below is a real decision made by Strands Decider 2B, scoring a closed set
      of four actions. Actions that are illegal in the current state are withdrawn before
      the model ever sees them &mdash; so the menu gets shorter as evidence accumulates,
      and what is left is always something the agent is actually allowed to do.
    </p>
  </header>

  <div class="cases" role="tablist" aria-label="Traces"></div>

  <section class="brief">
    <div>
      <p class="question"></p>
      <p class="caption"></p>
    </div>
    <div class="shot-slot"></div>
  </section>

  <section class="steps"></section>

  <section class="outcome">
    <span class="tag"></span>
    <p class="answer"></p>
  </section>

  <footer>
    Traces produced by <code>eval/run_benchmark.py</code> on the decider arm; nothing here
    is hand-written. Probabilities, confidence and latency are read straight from the
    model's response. <code>RoadSense &middot; Apache-2.0</code>
  </footer>
</div>

<script>
const TRACES = __TRACES__;
const ORDER = __ORDER__;
const LABEL = __LABELS__;

const $ = (s) => document.querySelector(s);
const tabs = $(".cases");
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function fmt(n, d = 3) { return n === null || n === undefined ? "—" : n.toFixed(d); }

function render(trace) {
  $(".question").textContent = trace.question;
  $(".caption").textContent = trace.caption;

  const slot = $(".shot-slot");
  slot.innerHTML = "";
  if (trace.image) {
    const img = document.createElement("img");
    img.className = "shot";
    img.src = trace.image;
    img.alt = "The photograph the agent was given";
    slot.append(img);
  } else {
    const box = document.createElement("div");
    box.className = "no-shot";
    box.textContent = "no image provided";
    slot.append(box);
  }

  const steps = $(".steps");
  steps.innerHTML = "";
  const fills = [];

  trace.steps.forEach((step) => {
    const el = document.createElement("div");
    el.className = "step";

    const head = document.createElement("div");
    head.className = "step-head";
    head.innerHTML =
      `<span class="step-n">Step ${step.n}</span>` +
      `<span class="step-pick">${LABEL[step.action]}</span>` +
      `<span class="menu-count">${step.offered.length} of 4 on the menu</span>`;
    el.append(head);

    const bars = document.createElement("div");
    bars.className = "bars";
    ORDER.forEach((action) => {
      const offered = step.offered.includes(action);
      const p = step.probs[action];
      const row = document.createElement("div");
      row.className =
        "bar-row" + (action === step.action ? " is-chosen" : "") + (offered ? "" : " withdrawn");
      row.innerHTML =
        `<span class="bar-label">${LABEL[action]}</span>` +
        `<span class="bar-track"><span class="bar-fill"></span></span>` +
        `<span class="bar-val">${offered ? fmt(p) : "withdrawn"}</span>`;
      bars.append(row);
      if (offered && p !== undefined) fills.push([row.querySelector(".bar-fill"), p]);
    });
    el.append(bars);

    const readouts = document.createElement("div");
    readouts.className = "readouts";
    readouts.innerHTML =
      `<span class="readout"><span class="readout-k">confidence</span>` +
      `<span class="readout-v">${fmt(step.confidence)}</span></span>` +
      `<span class="readout"><span class="readout-k">P(evidence sufficient)</span>` +
      `<span class="readout-v good">${fmt(step.sufficient)}</span></span>` +
      `<span class="readout"><span class="readout-k">latency</span>` +
      `<span class="readout-v">${step.latency} ms</span></span>`;
    el.append(readouts);

    if (step.rejected) {
      const note = document.createElement("div");
      note.className = "note";
      note.textContent = "Policy overruled this pick: " + step.rejected;
      el.append(note);
    }

    if (step.evidence) {
      const ev = document.createElement("div");
      const empty = !step.evidence.ok || step.evidence.quality === null;
      ev.className = "evidence" + (empty ? " empty" : "");
      const q = step.evidence.quality === null ? "no confidence reported"
                                               : "tool confidence " + fmt(step.evidence.quality, 2);
      ev.innerHTML =
        `<div class="ev-head"><span>${step.evidence.tool} returned</span><span>${q}</span></div>` +
        `<p class="ev-body"></p>`;
      ev.querySelector(".ev-body").textContent = step.evidence.summary;
      el.append(ev);
    }

    steps.append(el);
  });

  const tag = $(".tag");
  tag.textContent = trace.outcome.replace(/_/g, " ");
  tag.className = "tag" + (trace.outcome === "answered" ? "" : " insufficient");
  $(".answer").textContent = trace.answer;

  // One orchestrated reveal: the distributions fill in reading order.
  if (reduced) {
    fills.forEach(([node, p]) => { node.style.width = (p * 100) + "%"; });
  } else {
    fills.forEach(([node, p], i) => {
      setTimeout(() => { node.style.width = (p * 100) + "%"; }, 60 + i * 45);
    });
  }
}

TRACES.forEach((trace, i) => {
  const b = document.createElement("button");
  b.className = "case-btn";
  b.type = "button";
  b.setAttribute("role", "tab");
  b.textContent = trace.id + " · " + trace.label;
  b.addEventListener("click", () => {
    tabs.querySelectorAll(".case-btn").forEach((x) => x.setAttribute("aria-selected", "false"));
    b.setAttribute("aria-selected", "true");
    render(trace);
  });
  if (i === 0) b.setAttribute("aria-selected", "true");
  tabs.append(b);
});

render(TRACES[0]);
</script>
"""

LABELS_FOR_TAB = {
    "rs-021": "readable sign",
    "rs-030": "unreadable sign",
    "rs-037": "no image",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "eval/results/results_full.json"))
    ap.add_argument("--cases", nargs="+", default=["rs-021", "rs-030", "rs-037"])
    ap.add_argument("--out", default=str(ROOT / "demo/trace_viewer.html"))
    args = ap.parse_args()

    traces = collect(pathlib.Path(args.results), args.cases)
    for t in traces:
        t["label"] = LABELS_FOR_TAB.get(t["id"], t["id"])

    html = (
        PAGE.replace("__TRACES__", json.dumps(traces))
        .replace("__ORDER__", json.dumps(ACTION_ORDER))
        .replace("__LABELS__", json.dumps(ACTION_LABEL))
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(traces)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

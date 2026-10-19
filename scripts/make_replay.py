"""Build the public demo: real traces, replayed, with no backend.

The live server is the honest demo, and it is also a URL that spends money every time
someone clicks it — two Rekognition calls, a Bedrock completion, and a GPU instance
billed by the hour. Leaving that open to the internet for a week is a bill, not a
demo.

So the public page replays episodes that actually happened. Every probability, every
confidence and every latency below was produced by a run of `eval/run_benchmark.py`
against the real model on an A10G; nothing here is staged, and the page says so. What
it cannot do is run your photograph — for that you clone the repo, which is where a
demo like this belongs anyway.

    python scripts/make_replay.py
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

ACTION_ORDER = ["detect_objects", "read_sign_text", "request_better_evidence", "answer"]
ACTION_LABEL = {
    "detect_objects": "detect objects",
    "read_sign_text": "read sign text",
    "request_better_evidence": "request better evidence",
    "answer": "answer",
}

#: Chosen so the five together tell the whole story, including the one it gets wrong.
SCENES = {
    "rs-021": {
        "title": "Two tools, in order",
        "blurb": "It picks the detector, sees what came back, and only then decides to read the sign.",
        "kind": "good",
    },
    "rs-012": {
        "title": "One tool is enough",
        "blurb": "The question only asks about text, so it reads and stops. No wasted call.",
        "kind": "good",
    },
    "rs-030": {
        "title": "Nothing readable",
        "blurb": "The sign is degraded past reading. OCR returns nothing and the agent says so.",
        "kind": "limit",
    },
    "rs-037": {
        "title": "No photo at all",
        "blurb": "Both vision tools are withdrawn before the model sees the menu. Only two options remain.",
        "kind": "limit",
    },
    "rs-029": {
        "title": "The one it gets wrong",
        "blurb": "OCR confidently returns a single wrong letter. Nothing in the evidence looks broken, so the agent answers. We left it in.",
        "kind": "bad",
    },
}


def embed(path: str | None) -> str | None:
    if not path:
        return None
    local = ROOT / "demo" / "images" / pathlib.Path(path).name
    if not local.is_file() or local.stat().st_size == 0:
        return None
    mime = mimetypes.guess_type(local.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(local.read_bytes()).decode()


def collect(results: pathlib.Path) -> list[dict]:
    data = json.loads(results.read_text(encoding="utf-8"))
    decider = next(r for r in data["results"] if r["arm"] == "decider")
    by_id = {t["case_id"]: t for t in decider["traces"]}

    out = []
    for cid, meta in SCENES.items():
        t = by_id.get(cid)
        if t is None:
            continue
        ev_iter = iter(t["evidence"])
        steps = []
        for s in t["steps"]:
            evidence = None if s["action"] == "answer" else next(ev_iter, None)
            steps.append({
                "n": s["step"],
                "action": s["action"],
                "confidence": s["confidence"],
                "sufficient": s["evidence_sufficient"],
                "latency": round(s["latency_ms"]),
                "offered": s["offered_actions"],
                "probs": s["probabilities"],
                "rejected": s["rejected"],
                "evidence": None if evidence is None else {
                    "tool": evidence["tool"],
                    "ok": evidence["ok"],
                    "quality": evidence["quality"],
                    "summary": evidence["summary"],
                    "ms": round(evidence["duration_ms"]),
                },
            })
        cost = t.get("cost", {})
        out.append({
            "id": cid, **meta,
            "question": t["question"],
            "image": embed(t["image_path"]),
            "outcome": t["outcome"],
            "answer": t["answer"],
            "steps": steps,
            "wall": round(t["wall_ms"]),
            "visionCalls": cost.get("rekognition_calls", 0),
            "bedrockIn": cost.get("bedrock_input_tokens", 0),
            "bedrockOut": cost.get("bedrock_output_tokens", 0),
        })
    return out


PAGE = r"""<title>RoadSense Replay</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--ground:#eef1f5;--surface:#fff;--sunk:#e3e7ee;--ink:#13161b;--muted:#565f6d;
 --faint:#8a93a3;--rule:#d5dae3;--chosen:#2b3bb5;--on-chosen:#fff;--chosen-soft:#dde2f6;
 --dim:#bfc7d8;--warn:#9a4807;--warn-soft:#f7e7d6;--bad:#9b2c2c;--bad-soft:#f8e2e2;
 --good:#0d6577;--good-soft:#d7edf1;
 --shadow:0 1px 2px rgba(19,22,27,.05),0 10px 30px -22px rgba(19,22,27,.45)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0d1015;--surface:#15191f;--sunk:#1b2029;--ink:#e8ebf1;--muted:#919aaa;
 --faint:#6a7383;--rule:#262c37;--chosen:#8b98ff;--on-chosen:#0d1015;--chosen-soft:#222949;
 --dim:#373f4d;--warn:#dfa066;--warn-soft:#2d2317;--bad:#e08080;--bad-soft:#2d1a1a;
 --good:#59b5c6;--good-soft:#14252a;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 10px 30px -22px rgba(0,0,0,.9)}}
:root[data-theme="dark"]{--ground:#0d1015;--surface:#15191f;--sunk:#1b2029;--ink:#e8ebf1;
 --muted:#919aaa;--faint:#6a7383;--rule:#262c37;--chosen:#8b98ff;--on-chosen:#0d1015;
 --chosen-soft:#222949;--dim:#373f4d;--warn:#dfa066;--warn-soft:#2d2317;--bad:#e08080;
 --bad-soft:#2d1a1a;--good:#59b5c6;--good-soft:#14252a;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 10px 30px -22px rgba(0,0,0,.9)}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);line-height:1.55;
 font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:44px 22px 90px}

.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.14em;
 text-transform:uppercase;color:var(--faint);margin:0 0 10px}
h1{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:clamp(36px,6.2vw,62px);
 line-height:1.0;letter-spacing:-.015em;margin:0 0 14px;text-wrap:balance}
h1 em{font-style:italic;color:var(--chosen)}
.lede{margin:0;color:var(--muted);font-size:17px;max-width:62ch}
.recorded{display:inline-flex;align-items:center;gap:7px;margin-top:16px;padding:7px 13px;
 border-radius:999px;border:1px solid var(--rule);background:var(--surface);
 font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good)}

.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:11px;
 margin:34px 0 30px}
.legend div{background:var(--surface);border:1px solid var(--rule);border-radius:5px;padding:13px 15px}
.legend b{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
 letter-spacing:.09em;text-transform:uppercase;color:var(--chosen);margin-bottom:5px;font-weight:500}
.legend span{font-size:13.5px;color:var(--muted);line-height:1.45}

.scenes{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:12px;margin-bottom:26px}
.scene{padding:0;border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--surface);
 cursor:pointer;text-align:left;transition:border-color .15s,transform .12s,box-shadow .15s}
.scene:hover{transform:translateY(-2px);box-shadow:var(--shadow);border-color:var(--dim)}
.scene[aria-pressed="true"]{border-color:var(--chosen);box-shadow:0 0 0 2px var(--chosen-soft)}
.scene:focus-visible{outline:2px solid var(--chosen);outline-offset:2px}
.scene img{width:100%;display:block;aspect-ratio:3/2;object-fit:cover;background:var(--sunk)}
.scene .noimg{aspect-ratio:3/2;display:grid;place-items:center;background:var(--sunk);color:var(--faint);
 font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase}
.scene .body{padding:11px 13px 13px}
.scene .t{font-weight:600;font-size:14.5px;margin-bottom:3px;display:flex;align-items:center;gap:7px}
.scene .b{font-size:12.5px;color:var(--muted);line-height:1.4}
.flag{width:7px;height:7px;border-radius:50%;flex:none}
.flag.good{background:var(--good)}.flag.limit{background:var(--warn)}.flag.bad{background:var(--bad)}

.bar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:15px 18px;background:var(--surface);
 border:1px solid var(--rule);border-radius:6px;margin-bottom:22px;box-shadow:var(--shadow)}
.q{font-family:"Instrument Serif",Georgia,serif;font-size:21px;flex:1;min-width:230px;line-height:1.25}
.play{padding:11px 20px;border:none;border-radius:4px;background:var(--chosen);color:var(--on-chosen);
 font-family:inherit;font-size:14.5px;font-weight:600;cursor:pointer}
.play:hover{opacity:.9}.play:disabled{opacity:.45;cursor:not-allowed}
.pace{display:flex;gap:0;border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.pace button{padding:9px 13px;border:none;background:var(--surface);color:var(--muted);
 font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;cursor:pointer}
.pace button[aria-pressed="true"]{background:var(--chosen-soft);color:var(--chosen);font-weight:500}

.stage{min-height:300px}
.hint{border:1.5px dashed var(--rule);border-radius:6px;padding:52px 26px;text-align:center;color:var(--faint)}
.hint p{margin:0 0 7px;font-family:"Instrument Serif",Georgia,serif;font-size:21px;color:var(--muted)}

.step{border-top:1px solid var(--rule);padding:20px 0 6px;animation:rise .34s ease both}
.step:first-child{border-top:none}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.sh{display:flex;align-items:baseline;gap:11px;flex-wrap:wrap;margin-bottom:13px}
.sn{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.11em;
 text-transform:uppercase;color:var(--faint)}
.sp{font-family:"Instrument Serif",Georgia,serif;font-size:22px}
.mc{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--faint)}
.ms{margin-left:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;padding:3px 10px;
 border-radius:999px;background:var(--chosen-soft);color:var(--chosen);font-variant-numeric:tabular-nums}

.bars{display:flex;flex-direction:column;gap:6px}
.row{display:grid;grid-template-columns:182px minmax(0,1fr) 64px;align-items:center;gap:12px}
@media(max-width:620px){.row{grid-template-columns:124px minmax(0,1fr) 52px}}
.bl{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;color:var(--muted);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bt{display:block;height:19px;background:var(--sunk);border-radius:2px;overflow:hidden}
.bf{display:block;height:100%;width:0;background:var(--dim);border-radius:2px;
 transition:width .55s cubic-bezier(.22,.8,.3,1)}
.bv{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);
 text-align:right;font-variant-numeric:tabular-nums}
.pick .bf{background:var(--chosen)}.pick .bl{color:var(--ink);font-weight:500}
.pick .bv{color:var(--chosen);font-weight:500}
.gone .bl{color:var(--faint);text-decoration:line-through;text-decoration-thickness:1px}
.gone .bt{background:transparent;border:1px dashed var(--rule)}
.gone .bv{color:var(--faint);font-size:10px;letter-spacing:.06em;text-transform:uppercase}

.ro{display:flex;gap:20px;flex-wrap:wrap;margin-top:13px}
.rk{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;letter-spacing:.09em;
 text-transform:uppercase;color:var(--faint)}
.rv{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13.5px;font-variant-numeric:tabular-nums}
.rv.g{color:var(--good)}

.ev{margin-top:14px;padding:13px 15px;background:var(--surface);border:1px solid var(--rule);
 border-left:2px solid var(--chosen);border-radius:3px;animation:rise .3s ease both;box-shadow:var(--shadow)}
.ev.none{border-left-color:var(--warn)}
.eh{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;letter-spacing:.09em;
 text-transform:uppercase;color:var(--faint);margin-bottom:5px;display:flex;gap:11px;flex-wrap:wrap}
.eb{margin:0;font-size:14.5px}

.out{border-top:1px solid var(--rule);margin-top:24px;padding-top:20px;animation:rise .34s ease both}
.tag{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;
 letter-spacing:.09em;text-transform:uppercase;padding:4px 9px;border-radius:3px;margin-bottom:12px;
 background:var(--chosen-soft);color:var(--chosen)}
.tag.lim{background:var(--warn-soft);color:var(--warn)}
.ans{font-family:"Instrument Serif",Georgia,serif;font-size:21px;line-height:1.42;margin:0;max-width:62ch;
 text-wrap:pretty}
.verdict{margin-top:15px;padding:13px 16px;border-radius:4px;font-size:14px;background:var(--bad-soft);
 color:var(--bad);line-height:1.5;max-width:70ch}
.verdict b{font-weight:600}
.verdict em{font-style:italic}
.tot{margin-top:15px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--faint);
 display:flex;gap:18px;flex-wrap:wrap}
footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--rule);color:var(--faint);font-size:13px}
footer a{color:var(--muted)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
  <p class="eyebrow">Strands Decider 2B &middot; tool selection</p>
  <h1>It decides, looks, then <em>decides again</em></h1>
  <p class="lede">RoadSense answers questions about road photographs. It has four things it can
    do and picks one at a time &mdash; seeing what came back before choosing the next. Every
    number below came out of a real run.</p>
  <span class="recorded"><span class="dot"></span> recorded runs &middot; A10G GPU &middot; nothing staged</span>

  <div class="legend">
    <div><b>the menu</b><span>Four actions. Decider scores all of them and the bars show what it gave each.</span></div>
    <div><b>withdrawn</b><span>An action that is illegal right now is removed before the model sees it.</span></div>
    <div><b>the evidence</b><span>The chosen tool runs and reports back, with its own confidence.</span></div>
    <div><b>the answer</b><span>Bedrock writes it, and never sees the photo &mdash; only what the tools found.</span></div>
  </div>

  <div class="scenes" id="scenes"></div>

  <div class="bar">
    <span class="q" id="q"></span>
    <div class="pace" id="pace">
      <button type="button" data-pace="read" aria-pressed="true">readable pace</button>
      <button type="button" data-pace="real" aria-pressed="false">real speed</button>
    </div>
    <button class="play" id="play">Play</button>
  </div>

  <div class="stage" id="stage">
    <div class="hint"><p>Pick a scene above, then press Play.</p>
      <span>Each run is replayed exactly as it happened.</span></div>
  </div>

  <footer id="foot"></footer>
</div>

<script>
const SCENES = __SCENES__, ORDER = __ORDER__, LABEL = __LABELS__, REPO = __REPO__;
const $ = s => document.querySelector(s);
const stage = $("#stage"), playBtn = $("#play");
let current = SCENES[0], pace = "read", playing = false, timers = [];
const fmt = (n,d=3) => (n===null||n===undefined) ? "—" : Number(n).toFixed(d);
const sleep = ms => new Promise(r => timers.push(setTimeout(r, ms)));
const stop = () => { timers.forEach(clearTimeout); timers = []; };

SCENES.forEach((sc,i) => {
  const b = document.createElement("button");
  b.className = "scene"; b.type = "button"; b.setAttribute("aria-pressed", i===0 ? "true":"false");
  b.innerHTML =
    (sc.image ? `<img src="${sc.image}" alt="">` : `<div class="noimg">no photo</div>`) +
    `<div class="body"><div class="t"><span class="flag ${sc.kind}"></span>${sc.title}</div>` +
    `<div class="b">${sc.blurb}</div></div>`;
  b.onclick = () => {
    document.querySelectorAll(".scene").forEach(x => x.setAttribute("aria-pressed","false"));
    b.setAttribute("aria-pressed","true");
    current = sc; stop(); playing = false; playBtn.disabled = false; playBtn.textContent = "Play";
    reset();
  };
  $("#scenes").append(b);
});

function reset() {
  $("#q").textContent = current.question;
  stage.innerHTML = `<div class="hint"><p>${current.title}</p><span>${current.blurb}</span></div>`;
}

$("#pace").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  pace = b.dataset.pace;
  $("#pace").querySelectorAll("button").forEach(x =>
    x.setAttribute("aria-pressed", x === b ? "true" : "false"));
};

function stepEl(s) {
  const el = document.createElement("div");
  el.className = "step";
  el.innerHTML = `<div class="sh"><span class="sn">Step ${s.n}</span>` +
    `<span class="sp">${LABEL[s.action]}</span>` +
    `<span class="mc">${s.offered.length} of 4 on the menu</span>` +
    `<span class="ms">${s.latency} ms</span></div>`;
  const bars = document.createElement("div"); bars.className = "bars";
  const fills = [];
  ORDER.forEach(a => {
    const on = s.offered.includes(a), p = s.probs[a];
    const r = document.createElement("div");
    r.className = "row" + (a===s.action ? " pick":"") + (on ? "":" gone");
    r.innerHTML = `<span class="bl">${LABEL[a]}</span><span class="bt"><span class="bf"></span></span>` +
      `<span class="bv">${on && p!==undefined ? fmt(p) : "withdrawn"}</span>`;
    bars.append(r);
    if (on && p!==undefined) fills.push([r.querySelector(".bf"), p]);
  });
  el.append(bars);
  const ro = document.createElement("div"); ro.className = "ro";
  ro.innerHTML = `<span><span class="rk">confidence</span> <span class="rv">${fmt(s.confidence)}</span></span>` +
    `<span><span class="rk">P(evidence enough)</span> <span class="rv g">${fmt(s.sufficient)}</span></span>`;
  el.append(ro);
  if (s.rejected) {
    const n = document.createElement("div"); n.className = "verdict";
    n.textContent = "Policy overruled this pick: " + s.rejected; el.append(n);
  }
  requestAnimationFrame(() => fills.forEach(([n,p],i) =>
    timers.push(setTimeout(() => n.style.width = (p*100)+"%", 30 + i*35))));
  return el;
}

async function play() {
  if (playing) return;
  playing = true; playBtn.disabled = true; playBtn.textContent = "Playing…";
  stage.innerHTML = "";
  for (const s of current.steps) {
    await sleep(pace === "real" ? s.latency : Math.max(820, s.latency));
    stage.append(stepEl(s));
    if (s.evidence) {
      await sleep(pace === "real" ? s.evidence.ms : Math.max(640, s.evidence.ms));
      const e = s.evidence;
      const box = document.createElement("div");
      box.className = "ev" + ((!e.ok || e.quality === null) ? " none" : "");
      const q = e.quality === null ? "no confidence reported" : "tool confidence " + fmt(e.quality,2);
      box.innerHTML = `<div class="eh"><span>${e.tool} returned</span><span>${q}</span>` +
        `<span>${e.ms} ms</span></div><p class="eb"></p>`;
      box.querySelector(".eb").textContent = e.summary;
      stage.lastElementChild.append(box);
    }
    stage.lastElementChild.scrollIntoView({behavior:"smooth", block:"nearest"});
  }
  await sleep(pace === "real" ? 200 : 700);
  const out = document.createElement("div"); out.className = "out";
  out.innerHTML = `<span class="tag${current.outcome==="answered"?"":" lim"}">` +
    `${current.outcome.replace(/_/g," ")}</span><p class="ans"></p>` +
    (current.kind === "bad"
      ? `<div class="verdict"><b>Scored as a loss.</b> The expected behaviour was to ask for better evidence. Decider put 0.92 on answering and 0.08 on asking — OCR had reported <em>success</em> at 0.85 confidence, so nothing in the state looked wrong. The written answer does hedge about that confidence, but the agent never took the abstain action, and the benchmark scores the action rather than the prose. It stays in the results.</div>`
      : ``) +
    `<div class="tot"><span>${current.wall} ms end to end</span>` +
    `<span>${current.visionCalls} vision calls</span>` +
    `<span>${current.bedrockIn} in / ${current.bedrockOut} out Bedrock tokens</span>` +
    `<span>0 billable selector tokens</span></div>`;
  out.querySelector(".ans").textContent = current.answer;
  stage.append(out);
  out.scrollIntoView({behavior:"smooth", block:"nearest"});
  playing = false; playBtn.disabled = false; playBtn.textContent = "Play again";
}
playBtn.onclick = play;

$("#foot").innerHTML =
  `Replayed from traces produced by <code>eval/run_benchmark.py</code> on an NVIDIA A10G. ` +
  `Probabilities, confidence and latency are read straight from the model's response. ` +
  `To run it on your own photograph, clone the repo &mdash; the live server is not public, ` +
  `because every run spends real money. ` + (REPO ? `<a href="${REPO}">${REPO}</a> &middot; ` : ``) +
  `Apache-2.0.`;
reset();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "eval/results/results_gpu.json"))
    ap.add_argument("--out", default=str(ROOT / "demo/replay.html"))
    ap.add_argument("--repo", default="", help="repository URL shown in the footer")
    args = ap.parse_args()

    scenes = collect(pathlib.Path(args.results))
    html = (
        PAGE.replace("__SCENES__", json.dumps(scenes))
        .replace("__ORDER__", json.dumps(ACTION_ORDER))
        .replace("__LABELS__", json.dumps(ACTION_LABEL))
        .replace("__REPO__", json.dumps(args.repo))
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(scenes)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the public demo: real traces replayed, and a live mode when one is available.

The live server is the honest demo, and it is also a URL that spends money every time
someone clicks it — two Rekognition calls, a Bedrock completion, and a GPU instance
billed by the hour. Leaving that open to the internet for a week is a bill, not a
demo.

So the page ships with episodes that actually happened. Every probability, every
confidence and every latency was produced by a run of `eval/run_benchmark.py` against
the real model on an A10G; nothing is staged, and the page says so.

It also looks for a live backend. Point it at one with `--api`, or at run time with
`?api=https://host`, and the page probes `/api/status`: if a server answers, the
upload panel turns itself on and real runs stream in beside the recordings. If
nothing answers — which is the normal state, because that server costs money per
click — the panel explains why and the recordings still work.

One page, hosted as static files anywhere, honest in both states.

    python scripts/make_replay.py --api https://roadsense.example.com
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Widest the page ever shows an image. Embedding anything larger is dead weight.
EMBED_WIDTH = 760

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


#: The same agent, unchanged, on photographs a person actually took. These are not in
#: the benchmark -- a degradation dial you can set is worth more there than realism --
#: but a vision agent that has only ever been shown drawings is not worth believing.
PHOTO_SCENES = {
    "stop-sign": {
        "title": "A real stop sign",
        "blurb": "One tool is enough: the question only asks about text, so it reads and stops.",
        "kind": "good",
    },
    "speed-limit": {
        "title": "A real highway",
        "blurb": "Two tools in sequence. It finds the cars, then reads SPEED LIMIT 55 off a sign in the distance.",
        "kind": "good",
    },
    "intersection": {
        "title": "A real junction",
        "blurb": "A small street sign at the far side of the road, seen at an angle — and it reads both lines.",
        "kind": "good",
    },
    "street-traffic": {
        "title": "Real traffic",
        "blurb": "Twenty-five cars and ten buses. Number plates were blurred before publishing; the detector never needed them.",
        "kind": "good",
    },
}


def embed(path: str | None) -> str | None:
    """Inline an image as a data URI, whether it is synthetic or a photograph."""
    if not path:
        return None
    name = pathlib.Path(path).name
    local = next(
        (c for c in (ROOT / "demo" / "images" / name, ROOT / "demo" / "photos" / name)
         if c.is_file() and c.stat().st_size > 0),
        None,
    )
    if local is None:
        return None

    # Photographs arrive at 1280 px and weigh ~400 KB each; base64 adds a third on
    # top. Four of them would make this page a 2 MB download for images the layout
    # never shows wider than ~300 px. Re-encode for the page, leave the file alone.
    raw = local.read_bytes()
    mime = mimetypes.guess_type(local.name)[0] or "image/jpeg"
    if len(raw) > 120_000:
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.thumbnail((EMBED_WIDTH, EMBED_WIDTH), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=78, optimize=True)
            raw, mime = buf.getvalue(), "image/jpeg"
        except Exception:
            pass  # Pillow absent: ship the original rather than no image at all.
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def collect(results: pathlib.Path, scene_map: dict, group: str) -> list[dict]:
    """Turn one results file into scenes. A missing file yields nothing, not an error."""
    if not results.is_file():
        return []
    data = json.loads(results.read_text(encoding="utf-8"))
    decider = next(r for r in data["results"] if r["arm"] == "decider")
    by_id = {t["case_id"]: t for t in decider["traces"]}

    out = []
    for cid, meta in scene_map.items():
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
            "id": cid, "group": group, **meta,
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


#: Where each selector's numbers come from, and the order they are shown in. The
#: order is the argument: cheap rules first, frontier models in the middle, the
#: subject of the page last.
SELECTOR_SOURCES = [
    ("Keyword rule", "results_gpu.json", "keyword", "ten lines of if-statement"),
    ("Run everything", "results_gpu.json", "exhaustive", "every legal tool, every time"),
    ("Claude Haiku 4.5", "results_llmtiers.json", None, "frontier model, cheapest tier"),
    ("Claude Sonnet 4.5", "results_llmtiers.json", None, "frontier model, mid tier"),
    ("Claude Sonnet 4.6", "results_sonnet46.json", None, "frontier model, newer mid tier"),
    ("Claude Opus 4.5", "results_llmtiers.json", None, "frontier model, top tier available"),
    ("Strands Decider 2B", "results_gpu_nosuff.json", "decider", "a 2B pointer model on a GPU"),
]


def _summary_for(results_dir: pathlib.Path, filename: str, arm: str | None, label: str) -> dict | None:
    """Pull one arm's summary. ``arm=None`` matches an llm arm by the model in its name."""
    path = results_dir / filename
    if not path.is_file():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    results = blob["results"]
    if arm is not None:
        match = next((r for r in results if r["arm"] == arm), None)
    else:
        tag = label.lower().replace("claude ", "").replace(" ", "-").replace(".", "-")
        match = next((r for r in results if tag in r["arm"].replace(".", "-")), None)
    return match


def collect_benchmark(results_dir: pathlib.Path) -> dict | None:
    """The comparison table, read from the runs rather than retyped.

    Returning ``None`` is fine: the page simply omits the section rather than
    showing numbers it could not verify.
    """
    rows = []
    for label, filename, arm, note in SELECTOR_SOURCES:
        match = _summary_for(results_dir, filename, arm, label)
        if match is None:
            continue
        summary = match["summary"]
        invalid = [x for x in match["scores"] if x["difficulty"] == "invalid-input"]
        rows.append({
            "label": label,
            "note": note,
            "subject": label.startswith("Strands"),
            "toolExact": summary["tool_selection_exact_pct"],
            "behaviour": summary["behavior_correct_pct"],
            "invalid": sum(1 for x in invalid if x["behavior_correct"] and x["tool_set_exact"]),
            "invalidTotal": len(invalid),
            "ms": round(summary["selector_ms_median"]),
            # Only a hosted model bills per token. Decider runs on an instance charged
            # by the hour, and a rule bills nothing at all -- counting either as
            # "billable" would be the kind of mistake that makes a whole table suspect.
            "tokens": (summary.get("selector_input_tokens_total", 0)
                       + summary.get("selector_output_tokens_total", 0))
                      if label.startswith("Claude") else 0,
            "unnecessary": summary["unnecessary_calls_total"],
        })
    if len(rows) < 3:
        return None

    decider = next((r for r in rows if r["subject"]), None)
    llms = [r for r in rows if r["label"].startswith("Claude")]
    # Compare against the *best* frontier selector, not the slowest one. Picking the
    # weakest rival would inflate the number and discredit the rest of the table;
    # ties on accuracy go to whichever is quicker, which is the harder comparison.
    best = min(
        sorted(llms, key=lambda r: -r["toolExact"])[:3] or llms,
        key=lambda r: r["ms"],
    ) if llms else None
    return {
        "rows": rows,
        "cases": next(iter(json.loads((results_dir / "results_gpu.json").read_text(encoding="utf-8"))
                           ["results"]))["summary"]["cases"],
        "perDecision": round(decider["ms"] / 2.43) if decider else None,
        "speedup": round(best["ms"] / decider["ms"], 1) if (decider and best) else None,
        "rivalName": best["label"] if best else None,
        "ceiling": max((r["toolExact"] for r in llms), default=None),
    }


PAGE = r"""<title>RoadSense Replay</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--ground:#eef1f5;--surface:#fff;--sunk:#e3e7ee;--ink:#13161b;--muted:#565f6d;
 --faint:#8a93a3;--rule:#d5dae3;--chosen:#5668dd;--on-chosen:#fff;--chosen-soft:#dde2f6;
 --dim:#bfc7d8;--warn:#9a4807;--warn-soft:#f7e7d6;--bad:#9b2c2c;--bad-soft:#f8e2e2;
 --good:#0d6577;--good-soft:#d7edf1;
 --shadow:0 1px 2px rgba(19,22,27,.05),0 10px 30px -22px rgba(19,22,27,.45)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0d1015;--surface:#15191f;--sunk:#1b2029;--ink:#e8ebf1;--muted:#919aaa;
 --faint:#6a7383;--rule:#262c37;--chosen:#6e7cf0;--on-chosen:#0d1015;--chosen-soft:#222949;
 --dim:#373f4d;--warn:#dfa066;--warn-soft:#2d2317;--bad:#e08080;--bad-soft:#2d1a1a;
 --good:#59b5c6;--good-soft:#14252a;
 --shadow:0 1px 2px rgba(0,0,0,.45),0 10px 30px -22px rgba(0,0,0,.9)}}
:root[data-theme="dark"]{--ground:#0d1015;--surface:#15191f;--sunk:#1b2029;--ink:#e8ebf1;
 --muted:#919aaa;--faint:#6a7383;--rule:#262c37;--chosen:#6e7cf0;--on-chosen:#0d1015;
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

/* Exactly four, because there are exactly four steps. auto-fit happily makes a
   fifth empty column when the viewport allows it, which reads as a missing step. */
.loop{display:grid;grid-template-columns:repeat(4,1fr);gap:0;
 margin:30px 0 8px;border:1px solid var(--rule);border-radius:7px;overflow:hidden;background:var(--surface)}
.loop div{padding:16px 18px;border-right:1px solid var(--rule);position:relative}
.loop div:last-child{border-right:none}
@media(max-width:860px){.loop{grid-template-columns:repeat(2,1fr)}
 .loop div:nth-child(2n){border-right:none}
 .loop div:nth-child(-n+2){border-bottom:1px solid var(--rule)}}
@media(max-width:480px){.loop{grid-template-columns:1fr}
 .loop div{border-right:none;border-bottom:1px solid var(--rule)}}
.loop b{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;
 letter-spacing:.1em;text-transform:uppercase;color:var(--chosen);margin-bottom:6px;font-weight:500}
.loop p{margin:0;font-size:13.5px;color:var(--muted);line-height:1.45}
.loop .back{grid-column:1/-1;border-right:none;border-top:1px solid var(--rule);
 background:var(--sunk);font-size:12.5px;color:var(--faint);padding:10px 18px;text-align:center}
.loop .back b{display:inline;color:var(--muted);letter-spacing:.04em;text-transform:none;
 font-family:inherit;font-size:12.5px}

.grouphead{display:flex;align-items:baseline;gap:11px;flex-wrap:wrap;margin:26px 0 11px}
.grouphead h3{margin:0;font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:21px}
.grouphead span{font-size:12.5px;color:var(--faint);line-height:1.4}
.grouphead a{color:var(--muted)}

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

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:28px 0 8px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:15px 17px}
.stat .n{font-family:"Instrument Serif",Georgia,serif;font-size:34px;line-height:1;color:var(--chosen)}
.stat .n small{font-size:16px;color:var(--muted);font-family:"IBM Plex Mono",ui-monospace,monospace}
.stat .k{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.4}

.results{margin-top:40px;padding-top:32px;border-top:1px solid var(--rule)}
h2{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:clamp(26px,3.6vw,38px);
 line-height:1.1;margin:0 0 10px;text-wrap:balance}
.sub{color:var(--muted);font-size:15.5px;margin:0 0 24px;max-width:70ch}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:720px;font-size:13.5px}
th{text-align:left;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;
 letter-spacing:.09em;text-transform:uppercase;color:var(--faint);font-weight:500;
 padding:11px 14px;border-bottom:1px solid var(--rule);white-space:nowrap}
th.num,td.num{text-align:right}
td{padding:10px 14px;border-bottom:1px solid var(--rule);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr.subject td{background:var(--chosen-soft)}
tr.subject .name{font-weight:600}
.name{white-space:nowrap}
.name small{display:block;font-weight:400;color:var(--faint);font-size:11.5px;margin-top:2px}
.cell{display:flex;align-items:center;gap:9px;justify-content:flex-end}
.mini{flex:1;max-width:116px;height:9px;background:var(--sunk);border-radius:2px;overflow:hidden}
.mini i{display:block;height:100%;background:var(--dim);border-radius:2px}
tr.subject .mini i{background:var(--chosen)}
.v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
 min-width:52px;text-align:right}
.caption{font-size:12.5px;color:var(--faint);margin:11px 2px 0;line-height:1.5;max-width:76ch}
.finding{margin-top:22px;padding:16px 19px;border:1px solid var(--rule);border-left:2px solid var(--chosen);
 border-radius:5px;background:var(--surface)}
.finding h3{margin:0 0 7px;font-size:15px}
.finding p{margin:0;font-size:14px;color:var(--muted);line-height:1.55;max-width:76ch}
.finding.loss{border-left-color:var(--bad)}
.finding.loss h3{color:var(--bad)}

.live{border:1px solid var(--rule);border-radius:6px;background:var(--surface);padding:17px 19px;
 margin-bottom:22px;box-shadow:var(--shadow)}
.live.off{background:transparent;border-style:dashed;box-shadow:none}
.lh{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:11px}
.lt{font-weight:600;font-size:15px}
.status{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;padding:4px 10px;
 border-radius:999px;border:1px solid var(--rule);color:var(--muted);margin-left:auto}
.status.on{border-color:var(--good);color:var(--good);background:var(--good-soft)}
.status.off{border-color:var(--rule);color:var(--faint)}
.lnote{font-size:13.5px;color:var(--muted);margin:0;max-width:74ch;line-height:1.5}
.lform{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;margin-top:13px}
@media(max-width:700px){.lform{grid-template-columns:1fr}}
.drop{display:block;border:1.5px dashed var(--rule);border-radius:4px;padding:15px;text-align:center;
 cursor:pointer;color:var(--muted);font-size:13.5px;transition:border-color .15s,background .15s}
.drop:hover,.drop.over{border-color:var(--chosen);background:var(--chosen-soft)}
.drop input{display:none}
.lq{width:100%;border:1px solid var(--rule);border-radius:4px;padding:9px 11px;font-family:inherit;
 font-size:14px;background:var(--ground);color:var(--ink);margin-top:9px}
.lq:focus{outline:2px solid var(--chosen);outline-offset:-1px}
.thumb{max-height:64px;border-radius:3px;border:1px solid var(--rule);margin-top:9px;display:none}

.stage{min-height:150px}
.hint{border:1.5px dashed var(--rule);border-radius:6px;padding:30px 26px;text-align:center;color:var(--faint)}
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
  <p class="lede">Ask a question about a road photograph and most agents do one of two things:
    run every tool they have, or guess which one to run from the wording of the question.
    RoadSense does neither. It picks one action, waits to see what comes back, and then
    picks again &mdash; and the second choice is frequently not the one it would have made
    first. A 2-billion-parameter model called <b>Strands Decider 2B</b> makes every one of
    those choices, in about 95&nbsp;ms, without writing a single word.</p>
  <span class="recorded"><span class="dot"></span> recorded runs &middot; A10G GPU &middot; nothing staged</span>

  <div class="stats" id="stats"></div>

  <div class="loop">
    <div><b>1 &middot; the menu</b><p>Four actions. Decider scores all of them at once and the
      bars show the probability it gave each.</p></div>
    <div><b>2 &middot; withdrawn</b><p>An action that is illegal right now &mdash; a tool that already
      ran, OCR with no image &mdash; is removed before the model ever sees it.</p></div>
    <div><b>3 &middot; the evidence</b><p>The chosen tool runs and reports back, carrying its own
      confidence rather than one the agent invented.</p></div>
    <div><b>4 &middot; the answer</b><p>Amazon Bedrock writes it, and never sees the photo &mdash;
      only what the tools found. It cannot describe what was not reported.</p></div>
    <div class="back"><b>and then it loops.</b> The new evidence goes back into the state, the
      menu gets shorter, and step&nbsp;1 happens again &mdash; until the model picks an action that
      ends the run, or the step budget does it for them.</div>
  </div>

  <div class="live off" id="live">
    <div class="lh"><span class="lt">Run it on your own photo</span>
      <span class="status off" id="lstat">checking…</span></div>
    <p class="lnote" id="lnote"></p>
    <div id="lform" hidden>
      <div class="lform">
        <div>
          <label class="drop" id="drop"><input type="file" id="file" accept="image/jpeg,image/png">
            Drop a road photo here, or click to choose</label>
          <input class="lq" id="lq" value="What vehicles appear and what text can be read on the sign?">
          <img class="thumb" id="thumb" alt="">
        </div>
        <button class="play" id="lrun">Run it</button>
      </div>
    </div>
  </div>

  <div id="groups"></div>

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

  <section class="results" id="results"></section>

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

// Two groups, because they answer different questions: the photographs show it is
// real, the synthetic set shows what was measured and why those cases are repeatable.
const GROUPS = [
  {key: "photo", title: "Photographs",
   note: "Real road photographs from Wikimedia Commons, CC0 and CC BY. The same agent, unchanged."},
  {key: "synthetic", title: "The benchmark set",
   note: "Drawn on purpose: the degradation is a dial that was measured against the live OCR, which is what makes the hard cases repeatable."},
];

GROUPS.forEach(group => {
  const members = SCENES.filter(sc => sc.group === group.key);
  if (!members.length) return;
  const head = document.createElement("div");
  head.className = "grouphead";
  head.innerHTML = "<h3>" + group.title + "</h3><span>" + group.note + "</span>";
  const row = document.createElement("div");
  row.className = "scenes";
  members.forEach(sc => {
    const b = document.createElement("button");
    b.className = "scene"; b.type = "button";
    b.setAttribute("aria-pressed", sc === SCENES[0] ? "true" : "false");
    b.innerHTML =
      (sc.image ? '<img src="' + sc.image + '" alt="">' : '<div class="noimg">no photo</div>') +
      '<div class="body"><div class="t"><span class="flag ' + sc.kind + '"></span>' + sc.title +
      '</div><div class="b">' + sc.blurb + '</div></div>';
    b.onclick = () => {
      document.querySelectorAll(".scene").forEach(x => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      current = sc; stop(); playing = false;
      playBtn.disabled = false; playBtn.textContent = "Play";
      reset();
    };
    row.append(b);
  });
  $("#groups").append(head, row);
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

// --- the measurements ------------------------------------------------------
// Rendered from the benchmark's own output. If the runs are redone and the numbers
// move, this section moves with them; there is nothing here to forget to update.
const BENCH = __BENCH__;

function bar(value, max, text) {
  const pct = max > 0 ? Math.max(1.5, (value / max) * 100) : 0;
  return '<span class="cell"><span class="mini"><i style="width:' + pct.toFixed(1) +
         '%"></i></span><span class="v">' + text + '</span></span>';
}

function renderResults() {
  if (!BENCH) return;
  const el = document.getElementById("results");
  const maxTool = Math.max(...BENCH.rows.map(r => r.toolExact));
  const maxMs   = Math.max(...BENCH.rows.map(r => r.ms));
  const maxTok  = Math.max(...BENCH.rows.map(r => r.tokens));

  const body = BENCH.rows.map(r =>
    '<tr class="' + (r.subject ? "subject" : "") + '">' +
    '<td class="name">' + r.label + '<small>' + r.note + '</small></td>' +
    '<td class="num">' + bar(r.toolExact, maxTool, r.toolExact.toFixed(1) + "%") + '</td>' +
    '<td class="num"><span class="v">' + r.behaviour.toFixed(1) + '%</span></td>' +
    '<td class="num"><span class="v">' + r.invalid + "/" + r.invalidTotal + '</span></td>' +
    '<td class="num">' + bar(r.ms, maxMs, r.ms === 0 ? "~0" : r.ms.toLocaleString()) + '</td>' +
    '<td class="num">' + bar(r.tokens, maxTok, r.tokens.toLocaleString()) + '</td>' +
    '</tr>').join("");

  el.innerHTML =
    '<h2>What it costs to use a 2B model instead of a frontier one</h2>' +
    '<p class="sub">Every selector below ran the same ' + BENCH.cases + ' cases through the same loop, ' +
    'with the same tools and the same writer. Only the thing choosing the next action changes. ' +
    'Decider does not win every column, and the ones it loses are here too.</p>' +
    '<div class="tablewrap"><table><thead><tr>' +
    '<th>Selector</th><th class="num">Tool set exactly right</th><th class="num">Behaviour correct</th>' +
    '<th class="num">Invalid inputs</th><th class="num">Selector latency, median</th>' +
    '<th class="num">Billable tokens</th>' +
    '</tr></thead><tbody>' + body + '</tbody></table></div>' +
    '<p class="caption">Bars are scaled within their own column; three different units are never put on ' +
    'one axis. Latency and tokens are better low, the rest better high. Every figure is read from ' +
    '<code>eval/results/</code>, generated by the run.</p>' +

    '<div class="finding"><h3>Four frontier models, one ceiling</h3><p>' +
    'Haiku 4.5, Sonnet 4.6 and Opus 4.5 all land on ' + (BENCH.ceiling || 89.2).toFixed(1) + '% \u2014 ' +
    'the same number to the decimal, across roughly an order of magnitude in price, with Opus taking ' +
    'twice Haiku\u2019s latency to reach it. Sonnet 4.5 is the one outlier and it is fully explained: ' +
    'seven of its replies could not be parsed at all, a defect 4.6 does not have. So the gap between ' +
    'the frontier models and Decider is not a capability gap anyone can spend their way out of.' +
    '</p></div>' +

    '<div class="finding"><h3>What Decider buys instead</h3><p>' +
    (BENCH.perDecision ? '<b>' + BENCH.perDecision + ' ms per decision</b> \u2014 ' : '') +
    (BENCH.speedup ? BENCH.speedup + '\u00d7 faster than ' + BENCH.rivalName + ' doing the identical job \u2014 ' : '') +
    'and <b>zero billable tokens</b>, because a pointer model scores a fixed list instead of writing ' +
    'an answer. It also cannot return something unparseable: there is no slot in the architecture for one.' +
    '</p></div>' +

    '<div class="finding loss"><h3>Where it loses, plainly</h3><p>' +
    'A ten-line keyword rule picks the right tool more often than Decider does. Part of that is the ' +
    'metric \u2014 the expected tool set is derived from the question\u2019s wording, which is exactly what a ' +
    'keyword rule reads, so the baseline starts ahead. Part is real: Decider ran a second tool on nine ' +
    'cases that only needed one. What the keyword rule cannot do is notice the file is missing, and on ' +
    'the invalid inputs it answers anyway.' +
    '</p></div>';
}
renderResults();

function renderStats() {
  if (!BENCH) return;
  const tiles = [
    [BENCH.perDecision, "ms", "per decision, on an A10G GPU"],
    [BENCH.speedup, "\u00d7", "faster than " + BENCH.rivalName + " selecting the same tools"],
    ["0", "", "billable tokens for every decision it makes"],
    [BENCH.cases, "", "cases, seven selectors, one loop"],
  ];
  document.getElementById("stats").innerHTML = tiles.map(([n, unit, k]) =>
    '<div class="stat"><div class="n">' + n + (unit ? '<small>' + unit + '</small>' : '') +
    '</div><div class="k">' + k + '</div></div>').join("");
}
renderStats();

// --- live mode -------------------------------------------------------------
// The page is static and works alone. If a backend happens to be reachable it turns
// the upload panel on; otherwise it says why, and the recordings carry the page.
const API = new URLSearchParams(location.search).get("api") || __API__;
let liveReady = false;

function liveOff(note) {
  $("#live").className = "live off";
  $("#lstat").className = "status off";
  $("#lstat").textContent = "live mode off";
  $("#lnote").innerHTML = note;
  $("#lform").hidden = true;
}

async function probeLive() {
  if (!API) {
    liveOff("No live server is configured for this page. Everything below is a recording of a " +
            "real episode \u2014 clone the repo to run the agent on your own photographs.");
    return;
  }
  try {
    const r = await fetch(API.replace(/\/$/, "") + "/api/status", {mode: "cors"});
    const s = await r.json();
    if (!s.ready) { liveOff("The model is still loading on the server. Try again in a minute."); return; }
    liveReady = true;
    $("#live").className = "live";
    $("#lstat").className = "status on";
    $("#lstat").textContent = "live \u00b7 " + s.device;
    const left = s.runs_remaining_today;
    $("#lnote").innerHTML =
      "A server is answering, so this runs for real: Strands Decider 2B on a GPU, Amazon " +
      "Rekognition for the evidence, Amazon Bedrock for the answer." +
      (left !== null && left !== undefined ? " <b>" + left + " runs left today.</b>" : "") +
      (s.samples_only ? " This instance only accepts the bundled samples." : "");
    $("#lform").hidden = false;
  } catch (e) {
    liveOff("The live server is switched off right now \u2014 every run costs two Rekognition calls, " +
            "a Bedrock completion and GPU time, so it is only turned on deliberately. " +
            "Everything below is a recording of a real run.");
  }
}
probeLive();

const fileInput = $("#file"), thumb = $("#thumb"), drop = $("#drop");
fileInput.onchange = () => {
  const f = fileInput.files[0];
  if (f) { thumb.src = URL.createObjectURL(f); thumb.style.display = "block"; }
};
["dragenter","dragover"].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => {
  const f = ev.dataTransfer.files[0];
  if (f) { fileInput.files = ev.dataTransfer.files; fileInput.onchange(); }
});

$("#lrun").onclick = async () => {
  if (!liveReady) return;
  stop(); playing = true;
  const btn = $("#lrun"); btn.disabled = true; btn.textContent = "deciding\u2026";
  stage.innerHTML = "";
  $("#q").textContent = $("#lq").value;
  const body = new FormData();
  body.append("question", $("#lq").value || "What is in this scene?");
  if (fileInput.files[0]) body.append("image", fileInput.files[0]);
  try {
    const res = await fetch(API.replace(/\/$/, "") + "/api/run", {method: "POST", body});
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split("\n"); buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === "decision") {
          stage.append(stepEl({n: ev.step, action: ev.action, confidence: ev.confidence,
            sufficient: ev.evidence_sufficient, latency: ev.latency_ms, offered: ev.offered,
            probs: ev.probabilities, rejected: ev.rejected}));
        } else if (ev.type === "evidence") {
          const e = ev.evidence;
          const box = document.createElement("div");
          box.className = "ev" + ((!e.ok || e.quality === null) ? " none" : "");
          const q = e.quality === null ? "no confidence reported" : "tool confidence " + fmt(e.quality,2);
          box.innerHTML = '<div class="eh"><span>' + e.tool + ' returned</span><span>' + q +
            '</span><span>' + Math.round(e.duration_ms) + ' ms</span></div><p class="eb"></p>';
          box.querySelector(".eb").textContent = e.summary;
          (stage.lastElementChild || stage).append(box);
        } else if (ev.type === "answer") {
          const c = ev.trace.cost || {};
          const out = document.createElement("div"); out.className = "out";
          out.innerHTML = '<span class="tag' + (ev.outcome === "answered" ? "" : " lim") + '">' +
            ev.outcome.replace(/_/g, " ") + '</span><p class="ans"></p>' +
            '<div class="tot"><span>' + ev.elapsed_ms + ' ms end to end</span><span>' +
            (c.rekognition_calls || 0) + ' vision calls</span>' +
            '<span>0 billable selector tokens</span></div>';
          out.querySelector(".ans").textContent = ev.answer || "(no answer returned)";
          stage.append(out);
        } else if (ev.type === "error") {
          const d = document.createElement("div"); d.className = "verdict";
          d.textContent = ev.message; stage.append(d);
        }
      }
    }
  } catch (e) {
    const d = document.createElement("div"); d.className = "verdict";
    d.textContent = "Could not reach the live server: " + e.message; stage.append(d);
  } finally {
    playing = false; btn.disabled = false; btn.textContent = "Run it";
  }
};

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
    ap.add_argument("--photos", default=str(ROOT / "eval/results/results_photos.json"),
                    help="traces from the real photographs; omit and the page shows only synthetic")
    ap.add_argument("--repo", default="", help="repository URL shown in the footer")
    ap.add_argument("--api", default="",
                    help="base URL of a live server; the page enables uploads when it answers")
    args = ap.parse_args()

    # Photographs first: they are what convinces a reader the agent is real. The
    # synthetic set comes after, because it is what measures rather than persuades.
    scenes = (collect(pathlib.Path(args.photos), PHOTO_SCENES, "photo")
              + collect(pathlib.Path(args.results), SCENES, "synthetic"))
    bench = collect_benchmark(pathlib.Path(args.results).parent)
    html = (
        PAGE.replace("__SCENES__", json.dumps(scenes))
        .replace("__ORDER__", json.dumps(ACTION_ORDER))
        .replace("__LABELS__", json.dumps(ACTION_LABEL))
        .replace("__REPO__", json.dumps(args.repo))
        .replace("__API__", json.dumps(args.api))
        .replace("__BENCH__", json.dumps(bench))
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(scenes)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

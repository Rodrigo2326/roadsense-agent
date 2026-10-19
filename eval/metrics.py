"""Scoring one run, and aggregating many.

Every check here is deterministic and reads only the trace. Nothing is scored by
asking a model whether the answer was good -- that would make the headline numbers
depend on a judge nobody can audit.

The one place this costs us precision is *fidelity*. Checking that prose contains no
unsupported claim is not decidable with string matching, so the check is narrow and
its narrowness is stated: it catches quoted sign text that the OCR tool never
returned. That is the specific hallucination this agent exists to avoid, and it is
the one a reader of the report can verify by hand from the trace.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

#: Phrases that mark an answer as stating its own limit. Matched case-insensitively
#: against the answer text. Kept visible rather than buried so the report's
#: "expressed the limit" column can be checked against the raw answers.
LIMIT_PHRASES = (
    "cannot be", "can not be", "cannot confirm", "could not", "not confirmable",
    "no readable", "not readable", "unable to", "not possible", "no text",
    "insufficient", "not enough", "no image was provided", "would be needed",
    "is needed", "not determine", "cannot determine", "cannot identify",
    "cannot verify", "no vehicles", "not visible", "unclear", "too small",
    "higher-resolution", "clearer image", "not be reliably",
)

#: Only double quotes. Apostrophes in ordinary prose ("the sign's text") pair up with
#: each other and produce matches like 'the', which are noise rather than claims.
QUOTED = re.compile(r"[\"“]([A-Za-z0-9 |/-]{2,40})[\"”]")

#: Quoted strings that are not claims about what the image said.
QUOTE_STOPWORDS = {
    "detect_objects", "read_sign_text", "request_better_evidence", "answer",
}


def tools_run(trace: dict[str, Any]) -> list[str]:
    return [e["tool"] for e in trace["evidence"] if e["tool"] in ("detect_objects", "read_sign_text")]


def supported_tokens(trace: dict[str, Any]) -> set[str]:
    """Every word any tool actually produced, lowercased.

    Both tools count. Quoting a detector label ("Vehicle") is as grounded as quoting
    an OCR reading ("STOP"); an earlier version of this check looked only at OCR and
    flagged legitimate label quotes, which is the kind of false positive that makes a
    fidelity number worse than useless.
    """
    out: set[str] = set()
    for e in trace["evidence"]:
        for item in e.get("items") or []:
            for field in ("text", "name"):
                for word in re.findall(r"[A-Za-z0-9]+", str(item.get(field, ""))):
                    out.add(word.lower())
    return out


def ocr_tokens(trace: dict[str, Any]) -> set[str]:
    """Only what OCR returned. Used where the distinction matters."""
    out: set[str] = set()
    for e in trace["evidence"]:
        if e["tool"] != "read_sign_text":
            continue
        for item in e.get("items") or []:
            for word in str(item.get("text", "")).split():
                out.add(word.lower())
    return out


def expressed_limit(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in LIMIT_PHRASES)


def unsupported_quotes(trace: dict[str, Any]) -> list[str]:
    """Quoted strings in the answer whose words no tool ever produced.

    This is the fidelity check. It is narrow by design -- see the module docstring.
    """
    answer = trace.get("answer") or ""
    available = supported_tokens(trace)
    bad: list[str] = []
    for quoted in QUOTED.findall(answer):
        cleaned = quoted.strip()
        if cleaned.lower() in QUOTE_STOPWORDS:
            continue
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", cleaned)]
        if not words:
            continue
        if any(w not in available for w in words):
            bad.append(cleaned)
    return bad


def score_case(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    expected = set(case["expected_tools"])
    actual_list = tools_run(trace)
    actual = set(actual_list)

    behaviour_ok = (
        trace["outcome"] == "insufficient_evidence" or expressed_limit(trace.get("answer", ""))
        if case["expected_behavior"] == "insufficient"
        else trace["outcome"] == "answered"
    )

    bad_quotes = unsupported_quotes(trace)
    decider_steps = [s for s in trace["steps"] if s.get("decider_called")]

    return {
        "case_id": case["case_id"],
        "difficulty": case["difficulty"],
        "expected_behavior": case["expected_behavior"],
        "outcome": trace["outcome"],
        "tools_expected": sorted(expected),
        "tools_run": actual_list,
        "tool_set_exact": actual == expected,
        "unnecessary_calls": len(actual - expected),
        "missing_calls": len(expected - actual),
        "repeated_calls": len(actual_list) - len(actual),
        "behavior_correct": behaviour_ok,
        "expressed_limit": expressed_limit(trace.get("answer", "")),
        "unsupported_quotes": bad_quotes,
        "fidelity_ok": not bad_quotes,
        "steps": len(trace["steps"]),
        "decider_calls": len(decider_steps),
        "policy_overrides": trace.get("policy_overrides", 0),
        "stop_reason": trace.get("stop_reason", ""),
        "wall_ms": trace["wall_ms"],
        "selector_ms": round(sum(s["latency_ms"] for s in trace["steps"]), 1),
        "rekognition_calls": trace["cost"]["rekognition_calls"],
        "bedrock_input_tokens": trace["cost"]["bedrock_input_tokens"],
        "bedrock_output_tokens": trace["cost"]["bedrock_output_tokens"],
        "selector_input_tokens": trace["cost"].get("decider_input_tokens", 0),
        "selector_output_tokens": trace["cost"].get("decider_output_tokens", 0),
        "rekognition_usd": trace["cost"]["rekognition_usd"],
        "errored": trace["outcome"] == "error",
    }


def _pct(n: int, d: int) -> float | None:
    return None if d == 0 else round(100.0 * n / d, 1)


def _quantile(values: list[float], q: float) -> float | None:
    """p95 is only meaningful with enough samples; below 20 it is just the maximum."""
    if len(values) < 20:
        return None
    return round(statistics.quantiles(values, n=100)[int(q * 100) - 1], 1)


def aggregate(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scores)
    n = len(rows)
    if n == 0:
        return {"cases": 0}

    insufficient = [r for r in rows if r["expected_behavior"] == "insufficient"]
    answerable = [r for r in rows if r["expected_behavior"] == "answer"]
    walls = sorted(r["wall_ms"] for r in rows)
    selector = sorted(r["selector_ms"] for r in rows)

    return {
        "cases": n,
        "tool_selection_exact_pct": _pct(sum(r["tool_set_exact"] for r in rows), n),
        "unnecessary_calls_total": sum(r["unnecessary_calls"] for r in rows),
        "unnecessary_calls_per_case": round(sum(r["unnecessary_calls"] for r in rows) / n, 3),
        "missing_calls_total": sum(r["missing_calls"] for r in rows),
        "repeated_calls_total": sum(r["repeated_calls"] for r in rows),
        "behavior_correct_pct": _pct(sum(r["behavior_correct"] for r in rows), n),
        "correct_abstention_pct": _pct(
            sum(r["behavior_correct"] for r in insufficient), len(insufficient)
        ),
        "over_abstention_pct": _pct(
            sum(1 for r in answerable if not r["behavior_correct"]), len(answerable)
        ),
        "fidelity_ok_pct": _pct(sum(r["fidelity_ok"] for r in rows), n),
        "unsupported_quote_cases": [r["case_id"] for r in rows if not r["fidelity_ok"]],
        "completion_pct": _pct(sum(1 for r in rows if not r["errored"]), n),
        "error_pct": _pct(sum(r["errored"] for r in rows), n),
        "policy_overrides_total": sum(r["policy_overrides"] for r in rows),
        "budget_exhausted_cases": sum(1 for r in rows if r["stop_reason"] == "budget_exhausted"),
        "steps_per_case": round(sum(r["steps"] for r in rows) / n, 2),
        "selector_ms_median": round(statistics.median(selector), 1),
        "selector_ms_p95": _quantile(selector, 0.95),
        "wall_ms_median": round(statistics.median(walls), 1),
        "wall_ms_p95": _quantile(walls, 0.95),
        "rekognition_calls_total": sum(r["rekognition_calls"] for r in rows),
        "rekognition_usd_total": round(sum(r["rekognition_usd"] or 0 for r in rows), 4),
        "bedrock_input_tokens_total": sum(r["bedrock_input_tokens"] for r in rows),
        "bedrock_output_tokens_total": sum(r["bedrock_output_tokens"] for r in rows),
        # For the llm arm these are Bedrock tokens too, billed the same way as the
        # writer's. For the decider arm they are not billed per token at all -- the
        # model runs on an instance charged by the hour. Keeping the column separate
        # is what stops those two very different costs being added together.
        "selector_input_tokens_total": sum(r["selector_input_tokens"] for r in rows),
        "selector_output_tokens_total": sum(r["selector_output_tokens"] for r in rows),
    }

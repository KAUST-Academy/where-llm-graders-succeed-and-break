#!/usr/bin/env python3
"""closed_batch_grade.py -- grade the CV / ML exams with OpenAI or Anthropic
models through their Batch APIs (50% of standard price, results within 24 h).

The prompt is assembled by the SAME functions the paper's graders use
(`computer_vision_grade_with_local.build_prompt` /
`introduction_to_ai_grade_with_local.build_prompt`), so every request is
byte-identical to what the open-weights models saw: persona at character 0
of a single user message, no system prompt, reference solution + rubric
breakdown on, guidelines on for the CV exam only (the ML exam has none).

Sampling: the paper's Gemini runs are thinking-off, temperature 0. Here
OpenAI runs use `reasoning_effort` off/minimal and `temperature=0` when the
model accepts it; Anthropic's current models expose no sampling temperature
at all (thinking is disabled instead). The filename's `t` token records what
was ACTUALLY sent: `t00` only when temperature 0 was requested, `tvd`
("vendor default") otherwise, so the analysis never misfiles a run as a
greedy one. meta.json carries the full sampling record.

Stages (state lives in closed_runs/<run-id>/):

    smoke    one synchronous call, prints the parsed grade, usage and cost
    prepare  build every request (students x questions) -> requests.jsonl
    submit   upload / create the batch                    -> meta.json
    status   poll the batch(es)
    retry    resubmit every request that is missing, errored or truncated
    fetch    download all batches, write the run workbook in the standard
             ablation shape, write cost.json (real usage x price)

Example:
    python closed_batch_grade.py smoke   --exam cv --provider openai --model gpt-5.5 --persona neutral --student 1 --q 1
    python closed_batch_grade.py prepare --exam cv --provider openai --model gpt-5.5 --persona strict --run-id O02
    python closed_batch_grade.py submit  --run-id O02
    python closed_batch_grade.py status  --run-id O02 --wait
    python closed_batch_grade.py fetch   --run-id O02            # then: retry / fetch again if needed
    python closed_batch_grade.py fetch   --run-id O02 --publish  # copy into results/ + tracker row

Output workbook name follows the grid convention the analysis scripts parse:
    <RID>__<RID>_m-<model>_sol1_gd<0|1>_rs0_bd1_str-<persona>_<t00|tvd>_n1.xlsx
Without --publish it is written under closed_runs/<run-id>/ (pilot runs must
not leak into the analysis). --publish copies it into the exam's results/
directory AND appends a row to that exam's ablation_runs.xlsx tracker (the
ML analysis enumerates tracker rows, not files). NOTE: the CV analysis
currently classifies every non-`L-` run id as Gemini; extend
`_source_from_rid` / the family map there before publishing CV runs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RUNS_DIR = ROOT / "closed_runs"

EXAMS = {
    "cv": dict(module="computer_vision_grade_with_local", gd=1,
               results=ROOT / "computer_vision_results" / "results",
               tracker=ROOT / "computer_vision_results" / "ablation_runs.xlsx"),
    "ml": dict(module="introduction_to_ai_grade_with_local", gd=0,
               results=ROOT / "introduction_to_ai_results" / "results",
               tracker=ROOT / "introduction_to_ai_results" / "ablation_runs.xlsx"),
}

# USD per 1M tokens at STANDARD tier: (input, cached input, output).
# Batch API = 50% of these for both vendors. Prices fetched 2026-09-19 from
# developers.openai.com/api/docs/pricing and the Anthropic API rate card.
PRICES = {
    "gpt-5.5": (5.00, 0.50, 30.00), "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.6-sol": (4.00, 0.40, 20.00), "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5": (1.25, 0.125, 10.00), "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-4o": (2.50, 1.25, 10.00), "gpt-6-astra": (10.00, 1.00, 50.00),
    "claude-opus-5": (5.00, 0.50, 25.00), "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00), "claude-fable-5-1": (10.00, 0.25, 50.00),
}
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.0}   # Anthropic cache-write premiums by TTL

# Models that reject a disabled-thinking request (thinking is always on).
ANTHROPIC_THINKING_ALWAYS_ON = ("claude-fable", "claude-mythos")
# OpenAI models without a reasoning_effort knob at all.
OPENAI_NO_REASONING_KNOB = ("gpt-4o", "gpt-4.1", "gpt-3.5")

# JSON schema for the reply, mirroring SCHEMA_WITH_BREAKDOWN in the Gemini grader
# (lower-cased types for OpenAI). Non-strict: strict mode forbids maxLength.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "bonus": {"type": "number"},
        "task_breakdown": {
            "type": "array", "minItems": 1, "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "maxLength": 200},
                    "awarded": {"type": "number"},
                    "max": {"type": "number"},
                    "note": {"type": "string", "maxLength": 300},
                },
                "required": ["task", "awarded", "max"],
            },
        },
        "reasoning": {"type": "string", "maxLength": 1500},
    },
    "required": ["score", "bonus", "task_breakdown", "reasoning"],
}

STUDENT_MARKER = "## STUDENT SUBMISSION"
OK_FINISH = {"stop", "end_turn"}
RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,48}")   # + "-<student>-Q<q>" stays within 64


# ---------------------------------------------------------------------------
# environment / clients
# ---------------------------------------------------------------------------

def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def make_client(provider: str):
    env = load_env()
    if provider == "openai":
        import openai
        key = env.get("OPENAI_API_KEY") or env.get("GPT")
        if not key:
            sys.exit("no OpenAI key: put GPT=... in .env")
        return openai.OpenAI(api_key=key)
    if provider == "anthropic":
        import anthropic
        key = env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE")
        if not key:
            sys.exit("no Anthropic key: put CLAUDE=... in .env")
        return anthropic.Anthropic(api_key=key)
    raise ValueError(provider)


def price_for(model: str) -> tuple[float, float, float]:
    for k in sorted(PRICES, key=len, reverse=True):
        if model == k or model.startswith(k + "-") or model.startswith(k + "@"):
            return PRICES[k]
    raise KeyError(f"no price entry for {model}; add it to PRICES")


# ---------------------------------------------------------------------------
# exam module (the paper's grader) and prompt assembly
# ---------------------------------------------------------------------------

class Exam:
    def __init__(self, key: str):
        spec = EXAMS[key]
        self.key = key
        self.gd = spec["gd"]
        self.results_dir = spec["results"]
        self.tracker = spec["tracker"]
        self.m = importlib.import_module(spec["module"])
        self.questions = tuple(getattr(self.m, "QUESTIONS", (1, 2, 3, 4)))
        self.rubric = self.m.load_rubric_sections()
        self.guidelines = self.m.GUIDELINES.read_text(encoding="utf-8") if self.gd else ""
        self.template = self.m.GRADES_XLSX if key == "cv" else self.m.GRADES_CSV
        self.personas = sorted(self.m.STRICTNESS_PRESETS)

    def check_persona(self, persona: str) -> None:
        if persona not in self.m.STRICTNESS_PRESETS:
            sys.exit(f"unknown persona {persona!r}; choices: {self.personas}")

    def students(self) -> list[int]:
        return sorted(int(d.name) for d in self.m.EXTRACTED_DIR.iterdir()
                      if d.is_dir() and d.name.isdigit())

    def notebook(self, student: int, q: int) -> Optional[Path]:
        p = self.m.EXTRACTED_DIR / str(student) / f"Q{q}.ipynb"
        return p if p.exists() else None

    def prompt(self, student: int, q: int, persona: str) -> str:
        """Exactly what grade_student() builds for the default grid config."""
        self.check_persona(persona)
        nb = self.notebook(student, q)
        assert nb is not None
        sol = self.m.SOLUTIONS_DIR / self.m.QUESTION_META[q][2]
        return self.m.build_prompt(
            q=q,
            rubric_section=self.rubric[q],
            guidelines=self.guidelines,
            student_code=self.m.notebook_to_text(nb),
            solution_code=self.m.notebook_to_text(sol) if sol.exists() else None,
            use_breakdown=True,
            use_guidelines=bool(self.gd),
            strictness=persona,
            few_shot_examples=None,
        )

    def parse_reply(self, text: str) -> dict:
        t = text.strip()
        if t.startswith("```"):                      # strip a ```json fence
            t = t.split("\n", 1)[1] if "\n" in t else ""
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return self.m._parse_json_lenient(t)

    def reply_to_cells(self, q: int, result: dict) -> tuple[float, float, str]:
        """Clamp + breakdown-to-prose, identical to the graders; a reply with no
        numeric `score` is a failure, not a silent zero."""
        if result.get("score") is None:
            raise ValueError(f"no 'score' key (keys={sorted(result)[:8]})")
        raw_score = float(result.get("score", 0) or 0)
        raw_bonus = float(result.get("bonus", 0) or 0)
        q_max_score, q_max_bonus, _ = self.m.QUESTION_META[q]
        score = max(0.0, min(raw_score, q_max_score))
        bonus = max(0.0, min(raw_bonus, q_max_bonus))
        reasoning = result.get("reasoning", "") or ""
        breakdown = result.get("task_breakdown") or []
        if not isinstance(breakdown, list):
            breakdown = [breakdown]
        if breakdown:
            reasoning = reasoning + "\n\nBreakdown:\n" + "\n".join(
                f"- {b.get('task', '?')}: {b.get('awarded', '?')}/{b.get('max', '?')}"
                + (f" — {b['note']}" if b.get('note') else "")
                if isinstance(b, dict) else f"- {b}"
                for b in breakdown
            )
        return score, bonus, reasoning


def parse_student_list(spec: str, universe: list[int]) -> list[int]:
    """'1,2,5-10' -> sorted ids present in `universe`; warns about the rest."""
    wanted: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(tok))
    missing = sorted(wanted - set(universe))
    if missing:
        logging.warning("no extracted folder for students %s", missing)
    return [s for s in universe if s in wanted]


def split_for_cache(prompt: str) -> tuple[str, str]:
    """Shared prefix (persona+guidelines+rubric+solution) / student-specific rest."""
    i = prompt.find(STUDENT_MARKER)
    return (prompt, "") if i < 0 else (prompt[:i], prompt[i:])


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

def openai_body(prompt: str, model: str, max_out: int, reasoning: str, json_mode: str,
                temperature: Optional[float]) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_out,
    }
    if reasoning and reasoning != "default" and not model.startswith(OPENAI_NO_REASONING_KNOB):
        if model.startswith("gpt-6") and reasoning == "none":
            reasoning = "minimal"          # gpt-6 rejects 'none'
        body["reasoning_effort"] = reasoning
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode == "schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "grade", "schema": REPLY_SCHEMA, "strict": False}}
    elif json_mode == "object":
        body["response_format"] = {"type": "json_object"}
    return body


def anthropic_params(prompt: str, model: str, max_out: int, thinking_off: bool,
                     cache: bool, ttl: str) -> dict:
    prefix, rest = split_for_cache(prompt)
    if cache and rest:
        cc: dict[str, Any] = {"type": "ephemeral"}
        if ttl == "1h":
            cc["ttl"] = "1h"
        content = [
            {"type": "text", "text": prefix, "cache_control": cc},
            {"type": "text", "text": rest},
        ]
    else:
        content = [{"type": "text", "text": prompt}]
    params: dict[str, Any] = {
        "model": model, "max_tokens": max_out,
        "messages": [{"role": "user", "content": content}],
    }
    if thinking_off:
        if model.startswith(ANTHROPIC_THINKING_ALWAYS_ON):
            sys.exit(f"{model} cannot run with thinking disabled; pass --thinking-on "
                     f"(and raise --max-output-tokens, thinking shares max_tokens)")
        params["thinking"] = {"type": "disabled"}
    return params


def sampling_record(a) -> dict:
    """What we actually control on each provider; drives the filename's t token."""
    if a.provider == "openai":
        return {"temperature": a.temperature, "reasoning_effort": a.reasoning,
                "t_token": "t00" if a.temperature == 0 else "tvd"}
    return {"temperature": None, "note": "Anthropic exposes no sampling temperature on current models",
            "thinking": "on" if a.thinking_on else "disabled", "t_token": "tvd"}


def build_request(a, exam: Exam, cid: str, prompt: str) -> dict:
    if a.provider == "openai":
        return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                "body": openai_body(prompt, a.model, a.max_output_tokens, a.reasoning,
                                    a.json_mode, a.temperature)}
    return {"custom_id": cid,
            "params": anthropic_params(prompt, a.model, a.max_output_tokens,
                                       not a.thinking_on, not a.no_cache, a.cache_ttl)}


# ---------------------------------------------------------------------------
# usage -> cost
# ---------------------------------------------------------------------------

def normalize_usage(provider: str, usage: Any) -> dict:
    g = (lambda o, k: (o.get(k) if isinstance(o, dict) else getattr(o, k, None)))
    if provider == "openai":
        pd = g(usage, "prompt_tokens_details") or {}
        cd = g(usage, "completion_tokens_details") or {}
        cached = g(pd, "cached_tokens") or 0
        return {"input": (g(usage, "prompt_tokens") or 0) - cached, "cached": cached,
                "cache_write_5m": 0, "cache_write_1h": 0,
                "output": g(usage, "completion_tokens") or 0,
                "reasoning": g(cd, "reasoning_tokens") or 0}
    cc = g(usage, "cache_creation")
    w5 = (g(cc, "ephemeral_5m_input_tokens") or 0) if cc else 0
    w1 = (g(cc, "ephemeral_1h_input_tokens") or 0) if cc else 0
    total_w = g(usage, "cache_creation_input_tokens") or 0
    if not cc and total_w:                 # older response shape: assume 5m
        w5 = total_w
    return {"input": g(usage, "input_tokens") or 0,
            "cached": g(usage, "cache_read_input_tokens") or 0,
            "cache_write_5m": w5, "cache_write_1h": w1,
            "output": g(usage, "output_tokens") or 0, "reasoning": 0}


def cost_usd(model: str, u: dict, batch: bool) -> float:
    pi, pc, po = price_for(model)
    usd = (u["input"] * pi + u["cached"] * pc
           + u["cache_write_5m"] * pi * CACHE_WRITE_MULT["5m"]
           + u["cache_write_1h"] * pi * CACHE_WRITE_MULT["1h"]
           + u["output"] * po) / 1e6
    return usd * (0.5 if batch else 1.0)


def cache_net_usd(model: str, u: dict, batch: bool) -> float:
    """Positive = caching cost more than it saved (write premiums > read savings)."""
    pi, pc, _ = price_for(model)
    prem = (u["cache_write_5m"] * pi * (CACHE_WRITE_MULT["5m"] - 1)
            + u["cache_write_1h"] * pi * (CACHE_WRITE_MULT["1h"] - 1))
    saved = u["cached"] * (pi - pc)
    return (prem - saved) / 1e6 * (0.5 if batch else 1.0)


def empty_usage() -> dict:
    return {"input": 0, "cached": 0, "cache_write_5m": 0, "cache_write_1h": 0, "output": 0, "reasoning": 0}


# ---------------------------------------------------------------------------
# run state
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_meta(run_id: str) -> dict:
    p = run_dir(run_id) / "meta.json"
    if not p.exists():
        sys.exit(f"{p} missing: run `prepare` first")
    return json.loads(p.read_text())


def save_meta(run_id: str, meta: dict) -> None:
    (run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2))


def custom_id(run_id: str, student: int, q: int) -> str:
    return f"{run_id}-{student}-Q{q}"      # [A-Za-z0-9_-] only (Anthropic rule)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------------
# provider batch plumbing
# ---------------------------------------------------------------------------

def submit_batch(client, provider: str, requests_path: Path, run_id: str) -> dict:
    if provider == "openai":
        with open(requests_path, "rb") as fh:
            f = client.files.create(file=fh, purpose="batch")
        b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h", metadata={"run_id": run_id})
        return {"batch_id": b.id, "input_file_id": f.id, "requests_file": requests_path.name,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": b.status}
    reqs = read_jsonl(requests_path)
    b = client.messages.batches.create(requests=reqs)
    return {"batch_id": b.id, "requests_file": requests_path.name,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": b.processing_status}


def batch_status(client, provider: str, batch_id: str) -> tuple[bool, str]:
    if provider == "openai":
        b = client.batches.retrieve(batch_id)
        rc = b.request_counts
        s = f"{b.status}  done={rc.completed}/{rc.total} failed={rc.failed}"
        if b.status in ("failed", "expired", "cancelled"):
            s += f"  errors={b.errors}"
        return b.status in ("completed", "failed", "expired", "cancelled"), s
    b = client.messages.batches.retrieve(batch_id)
    rc = b.request_counts
    s = (f"{b.processing_status}  processing={rc.processing} succeeded={rc.succeeded} "
         f"errored={rc.errored} canceled={rc.canceled} expired={rc.expired}")
    return b.processing_status == "ended", s


def download_results(client, provider: str, batch_id: str) -> list[dict]:
    """One normalized row per result: custom_id, ok, text, usage, finish, error, model."""
    rows: list[dict] = []
    if provider == "openai":
        b = client.batches.retrieve(batch_id)
        for fid in (b.output_file_id, b.error_file_id):
            if not fid:
                continue
            for line in client.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                resp = r.get("response") or {}
                body = resp.get("body") or {}
                ok = resp.get("status_code") == 200 and not r.get("error")
                text, finish, err, model = "", None, r.get("error"), body.get("model")
                if ok:
                    ch = (body.get("choices") or [{}])[0]
                    msg = ch.get("message") or {}
                    text = msg.get("content") or ""
                    finish = ch.get("finish_reason")
                    if not text:
                        ok, err = False, {"empty_content": True, "refusal": msg.get("refusal"),
                                          "finish_reason": finish}
                else:
                    err = err or body
                rows.append({"custom_id": r["custom_id"], "ok": ok, "text": text,
                             "usage": normalize_usage("openai", body.get("usage") or {}) if body.get("usage") else None,
                             "finish": finish, "error": err, "model": model, "batch_id": batch_id})
        return rows
    for r in client.messages.batches.results(batch_id):
        ok = r.result.type == "succeeded"
        text, usage, finish, err, model = "", None, None, None, None
        if ok:
            msg = r.result.message
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            usage = normalize_usage("anthropic", msg.usage)
            finish, model = msg.stop_reason, msg.model
            if not text:
                ok, err = False, {"empty_content": True, "stop_reason": finish,
                                  "stop_details": str(getattr(msg, "stop_details", None))}
        else:
            err = r.result.model_dump() if hasattr(r.result, "model_dump") else str(r.result)
        rows.append({"custom_id": r.custom_id, "ok": ok, "text": text, "usage": usage,
                     "finish": finish, "error": err, "model": model, "batch_id": batch_id})
    return rows


def usable(row: Optional[dict]) -> bool:
    """A result we will score: succeeded, non-empty, and not cut off."""
    return bool(row) and row["ok"] and (row["finish"] in OK_FINISH)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_smoke(a) -> None:
    exam = Exam(a.exam)
    exam.check_persona(a.persona)
    if exam.notebook(a.student, a.q) is None:
        sys.exit(f"student {a.student} has no Q{a.q} notebook")
    prompt = exam.prompt(a.student, a.q, a.persona)
    client = make_client(a.provider)
    req = build_request(a, exam, custom_id("SMOKE", a.student, a.q), prompt)
    t0 = time.time()
    if a.provider == "openai":
        resp = client.chat.completions.create(**req["body"])
        text = resp.choices[0].message.content or ""
        usage = normalize_usage("openai", resp.usage)
        finish, served = resp.choices[0].finish_reason, resp.model
    else:
        resp = client.messages.create(**req["params"])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = normalize_usage("anthropic", resp.usage)
        finish, served = resp.stop_reason, resp.model
    dt = time.time() - t0
    print(f"model={served}  finish={finish}  {dt:.1f}s  prompt_chars={len(prompt)}  "
          f"sampling={sampling_record(a)}")
    if finish not in OK_FINISH:
        print(f"!!! finish={finish}: this reply would be treated as FAILED in a batch "
              f"(raise --max-output-tokens or check refusals) !!!")
    print(f"usage={usage}  cost@standard=${cost_usd(a.model, usage, False):.4f}  "
          f"cost@batch=${cost_usd(a.model, usage, True):.4f}")
    result = exam.parse_reply(text)
    score, bonus, reasoning = exam.reply_to_cells(a.q, result)
    print(f"score={score} bonus={bonus} breakdown_entries={len(result.get('task_breakdown') or [])}")
    print("reasoning:", reasoning[:500].replace("\n", " | "))
    out = run_dir(f"smoke-{a.provider}") / f"{a.exam}-{a.model}-{a.persona}-s{a.student}-Q{a.q}.json"
    out.write_text(json.dumps({"prompt_chars": len(prompt), "request": req, "usage": usage,
                               "finish": finish, "model": served, "text": text, "parsed": result},
                              indent=2))
    print("saved", out)


def cmd_prepare(a) -> None:
    if not RUN_ID_RE.fullmatch(a.run_id):
        sys.exit("--run-id must match [A-Za-z0-9_-]{1,48}")
    mp = run_dir(a.run_id) / "meta.json"
    if mp.exists() and json.loads(mp.read_text()).get("batches"):
        sys.exit(f"{a.run_id} was already submitted; pick a new --run-id or delete {mp}")
    exam = Exam(a.exam)
    exam.check_persona(a.persona)
    if a.provider == "anthropic" and not a.thinking_on and a.model.startswith(ANTHROPIC_THINKING_ALWAYS_ON):
        sys.exit(f"{a.model} needs --thinking-on")
    students = exam.students()
    if a.students:
        students = parse_student_list(a.students, students)
    if a.limit:
        students = students[: a.limit]
    reqs, skipped = [], 0
    for s in students:
        for q in exam.questions:
            if exam.notebook(s, q) is None:
                skipped += 1
                continue
            reqs.append(build_request(a, exam, custom_id(a.run_id, s, q), exam.prompt(s, q, a.persona)))
    d = run_dir(a.run_id)
    write_jsonl(d / "requests.jsonl", reqs)
    chars = sum(len(json.dumps(r)) for r in reqs)
    meta = dict(run_id=a.run_id, exam=a.exam, provider=a.provider, model=a.model, persona=a.persona,
                config=f"sol1_gd{exam.gd}_rs0_bd1", sampling=sampling_record(a),
                students=students, n_requests=len(reqs), n_missing_notebooks=skipped,
                reasoning=a.reasoning, json_mode=a.json_mode, thinking_on=a.thinking_on,
                cache=not a.no_cache, cache_ttl=a.cache_ttl, max_output_tokens=a.max_output_tokens,
                prepared_at=time.strftime("%Y-%m-%d %H:%M:%S"), batches=[])
    save_meta(a.run_id, meta)
    print(f"{a.run_id}: {len(reqs)} requests for {len(students)} students "
          f"({skipped} missing notebooks skipped), {chars/1e6:.1f} MB -> {d/'requests.jsonl'}")


def cmd_submit(a) -> None:
    meta = load_meta(a.run_id)
    if meta.get("batches"):
        sys.exit(f"{a.run_id} already submitted ({[b['batch_id'] for b in meta['batches']]}); use `retry`")
    client = make_client(meta["provider"])
    b = submit_batch(client, meta["provider"], run_dir(a.run_id) / "requests.jsonl", a.run_id)
    meta["batches"].append(b)
    save_meta(a.run_id, meta)
    print(f"submitted {meta['provider']} batch {b['batch_id']} ({meta['n_requests']} requests) status={b['status']}")


def cmd_status(a) -> None:
    meta = load_meta(a.run_id)
    if not meta.get("batches"):
        sys.exit("not submitted yet")
    client = make_client(meta["provider"])
    t0 = time.time()
    while True:
        all_done = True
        for b in meta["batches"]:
            done, s = batch_status(client, meta["provider"], b["batch_id"])
            all_done &= done
            print(f"[{time.strftime('%H:%M:%S')}] {a.run_id} {b['batch_id']}: {s}")
        if all_done or not a.wait or time.time() - t0 > a.timeout:
            break
        time.sleep(a.interval)


def collect_results(meta: dict, client) -> dict[str, dict]:
    """Download every finished batch; later batches override earlier rows."""
    d = run_dir(meta["run_id"])
    by_id: dict[str, dict] = {}
    for b in meta["batches"]:
        done, s = batch_status(client, meta["provider"], b["batch_id"])
        if not done:
            print(f"skipping unfinished batch {b['batch_id']}: {s}")
            continue
        for r in download_results(client, meta["provider"], b["batch_id"]):
            prev = by_id.get(r["custom_id"])
            if prev is None or usable(r) or not usable(prev):
                by_id[r["custom_id"]] = r
    write_jsonl(d / "results.jsonl", list(by_id.values()))
    return by_id


def cmd_retry(a) -> None:
    meta = load_meta(a.run_id)
    client = make_client(meta["provider"])
    by_id = collect_results(meta, client)
    reqs = read_jsonl(run_dir(a.run_id) / "requests.jsonl")
    todo = [r for r in reqs if not usable(by_id.get(r["custom_id"]))]
    if not todo:
        print("nothing to retry: every request has a usable result")
        return
    if a.max_output_tokens:
        for r in todo:                     # e.g. bump the cap for truncated replies
            if meta["provider"] == "openai":
                r["body"]["max_completion_tokens"] = a.max_output_tokens
            else:
                r["params"]["max_tokens"] = a.max_output_tokens
    n = len(meta["batches"])
    p = run_dir(a.run_id) / f"requests.retry{n}.jsonl"
    write_jsonl(p, todo)
    reasons = {}
    for r in todo:
        row = by_id.get(r["custom_id"])
        k = "missing" if row is None else (row["finish"] if row["ok"] else "error")
        reasons[k] = reasons.get(k, 0) + 1
    print(f"retrying {len(todo)} requests ({reasons})")
    if a.dry_run:
        print("dry run: not submitted;", p)
        return
    b = submit_batch(client, meta["provider"], p, a.run_id)
    meta["batches"].append(b)
    save_meta(a.run_id, meta)
    print(f"submitted retry batch {b['batch_id']} status={b['status']}")


def cmd_fetch(a) -> None:
    meta = load_meta(a.run_id)
    d = run_dir(a.run_id)
    client = make_client(meta["provider"])
    by_id = collect_results(meta, client)
    rows = list(by_id.values())
    n_ok = sum(usable(r) for r in rows)
    n_missing = meta["n_requests"] - len(rows)
    print(f"{len(rows)} results ({n_ok} usable, {len(rows) - n_ok} failed/truncated, "
          f"{n_missing} missing) -> {d/'results.jsonl'}")

    tot = empty_usage()
    for r in rows:
        if r["usage"]:
            for k in tot:
                tot[k] += r["usage"].get(k, 0)
    cost = {"tokens": tot, "usd_batch": round(cost_usd(meta["model"], tot, True), 2),
            "usd_standard_equiv": round(cost_usd(meta["model"], tot, False), 2),
            "cache_net_usd_batch": round(cache_net_usd(meta["model"], tot, True), 2),
            "n_results": len(rows), "n_usable": n_ok,
            "per_call_usd_batch": round(cost_usd(meta["model"], tot, True) / max(len(rows), 1), 4),
            "models_served": sorted({r["model"] for r in rows if r.get("model")})}
    (d / "cost.json").write_text(json.dumps(cost, indent=2))
    print("cost:", cost)
    if a.publish and (n_ok < meta["n_requests"]):
        sys.exit(f"refusing --publish: {meta['n_requests'] - n_ok} requests lack a usable result; run `retry`")
    write_workbook(meta, by_id, cost, publish=a.publish)


def write_workbook(meta: dict, by_id: dict[str, dict], cost: dict, publish: bool) -> None:
    exam = Exam(meta["exam"])
    m = exam.m
    import openpyxl
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    rid, model, persona = meta["run_id"], meta["model"], meta["persona"]
    t_tok = meta["sampling"]["t_token"]
    name = f"{rid}__{rid}_m-{model}_{meta['config']}_str-{persona}_{t_tok}_n1.xlsx"
    dest = run_dir(rid) / name
    if dest.exists():
        dest.unlink()
    m.init_output_workbook(exam.template, dest)
    wb = openpyxl.load_workbook(dest)
    ws = wb.active
    cols = m.ensure_ai_columns(ws)
    row_index = m.build_student_row_index(ws)
    n_students = n_failed_q = n_parse_fail = n_trunc = 0
    for s in meta["students"]:
        row = row_index.get(s)
        if row is None:
            logging.warning("student %s has no row in the grades sheet; skipped", s)
            continue
        totals_score = totals_bonus = 0.0
        graded_any, failed = False, []
        for q in exam.questions:
            score_col = cols[f"AI Q{q} Score"]
            reason_col = cols[f"AI Q{q} Reasoning"]
            bonus_col = cols.get(f"AI Q{q} Bonus")
            if exam.notebook(s, q) is None:
                ws.cell(row=row, column=score_col, value=0)
                if bonus_col:
                    ws.cell(row=row, column=bonus_col, value=0)
                ws.cell(row=row, column=reason_col, value="No submission found.")
                continue
            r = by_id.get(custom_id(rid, s, q))
            if not r or not r["ok"]:
                failed.append(q)
                continue
            if r["finish"] not in OK_FINISH:
                logging.warning("student %s Q%d: finish=%s; treated as failed", s, q, r["finish"])
                n_trunc += 1
                failed.append(q)
                continue
            try:
                result = exam.parse_reply(r["text"])
                score, bonus, reasoning = exam.reply_to_cells(q, result)
            except Exception as e:  # noqa: BLE001
                logging.warning("student %s Q%d: unusable reply (%s): %.120s", s, q, e, r["text"])
                n_parse_fail += 1
                failed.append(q)
                continue
            ws.cell(row=row, column=score_col, value=score)
            if bonus_col:
                ws.cell(row=row, column=bonus_col, value=bonus)
            # models occasionally emit control characters XML forbids; openpyxl
            # raises IllegalCharacterError on them, so strip rather than crash
            ws.cell(row=row, column=reason_col, value=ILLEGAL_CHARACTERS_RE.sub("", reasoning))
            totals_score += score
            totals_bonus += bonus
            graded_any = True
        if graded_any:
            n_students += 1
            if failed:               # same rule as the graders: no partial totals
                n_failed_q += len(failed)
                ws.cell(row=row, column=cols["AI Total Score"]).value = None
                ws.cell(row=row, column=cols["AI Total Bonus"]).value = None
            else:
                ws.cell(row=row, column=cols["AI Total Score"], value=totals_score)
                ws.cell(row=row, column=cols["AI Total Bonus"], value=totals_bonus)
    m.atomic_save(wb, dest)
    summary = summarize(exam, dest)
    print(f"workbook: {dest}\n  students with grades={n_students} failed_questions={n_failed_q} "
          f"truncated={n_trunc} unparseable={n_parse_fail}\n  summary={summary}")
    meta.update(workbook=str(dest), summary=summary, cost=cost,
                n_failed_questions=n_failed_q, n_truncated=n_trunc, n_unparseable=n_parse_fail)
    if publish:
        import shutil
        out = exam.results_dir / name
        shutil.copy2(dest, out)
        meta["published"] = str(out)
        print("published ->", out)
        add_tracker_row(exam, meta, name, summary)
    save_meta(rid, meta)


def add_tracker_row(exam: Exam, meta: dict, name: str, summary: dict) -> None:
    """Append one row to the exam's ablation_runs.xlsx (the ML analysis
    enumerates tracker rows; the CV one reads thinking_mode counts from it)."""
    import openpyxl
    tracker = exam.tracker
    if not tracker.exists():
        logging.warning("tracker %s missing; no row added", tracker)
        return
    with exam.m._file_lock(tracker):
        wb = openpyxl.load_workbook(tracker)
        ws = wb.active
        headers = [c.value for c in ws[1]]
        if meta["run_id"] in {r[0].value for r in ws.iter_rows(min_row=2)}:
            logging.warning("tracker already has run_id %s; not duplicated", meta["run_id"])
            return
        run_tag = name[:-5].split("__", 1)[1]
        vals = {
            "run_id": meta["run_id"], "side": "closed",
            "hypothesis": f"{meta['provider']} {meta['model']} + {meta['persona']} (batch API)",
            "model": meta["model"], "use_solution": 1, "use_guidelines": exam.gd, "use_reasoning": 0,
            "use_breakdown": 1, "strictness": meta["persona"],
            "temperature": meta["sampling"].get("temperature"), "runs": 1,
            "max_output_tokens": meta["max_output_tokens"], "few_shot": 0,
            "students": "all" if len(meta["students"]) == len(exam.students()) else ",".join(map(str, meta["students"])),
            "thinking_mode": "off" if (meta["provider"] == "anthropic" and not meta["thinking_on"]) or
                             (meta["provider"] == "openai" and meta["reasoning"] in ("none", "minimal")) else "on",
            "run_tag": run_tag, "output_xlsx": f"results/{name}",
            "command": f"python closed_batch_grade.py prepare --exam {meta['exam']} --provider {meta['provider']} "
                       f"--model {meta['model']} --persona {meta['persona']} --run-id {meta['run_id']}; submit; fetch --publish",
            "status": "done", "started_at": meta["batches"][0]["submitted_at"] if meta.get("batches") else "",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "MAE_vs_TA_avg": summary.get("MAE", ""), "n_graded": summary.get("n", 0),
            "notes": f"batch ids {[b['batch_id'] for b in meta.get('batches', [])]}; "
                     f"usd_batch={meta.get('cost', {}).get('usd_batch')}; sampling={meta['sampling']}",
        }
        ws.append([vals.get(h, None) for h in headers])
        exam.m.atomic_save(wb, tracker)
    print("tracker row added ->", tracker)


def summarize(exam: Exam, xlsx: Path) -> dict:
    """MAE / bias of the AI total against the grader average, on the exam's own
    convention (CV: base score only; ML: score + bonus vs. the graders' totals)."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True).active
    hdr = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(hdr) if h}
    if exam.key == "cv":
        t1, t2 = col["TA 1 - Total Score (out of 35)"], col["TA 2 - Total Score (out of 35)"]
        ai_cols = [col["AI Total Score"]]
    else:
        t1, t2 = col["TA 1 - Total Grade"], col["TA 2 - Total Grade"]
        ai_cols = [col["AI Total Score"], col["AI Total Bonus"]]
    diffs = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        vals = [r[c] for c in ai_cols]
        if any(v in (None, "") for v in vals):
            continue
        ai = sum(float(v) for v in vals)
        ta = [r[t1], r[t2]]
        if any(t in (None, "") for t in ta):
            continue
        diffs.append(ai - (float(ta[0]) + float(ta[1])) / 2)
    if not diffs:
        return {"n": 0}
    return {"n": len(diffs), "MAE": round(statistics.fmean(abs(d) for d in diffs), 3),
            "bias": round(statistics.fmean(diffs), 3)}


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_run(sp):
        sp.add_argument("--run-id", required=True, help="e.g. O01 ([A-Za-z0-9_-], <=48 chars)")

    def gen(sp):
        sp.add_argument("--exam", choices=list(EXAMS), required=True)
        sp.add_argument("--provider", choices=["openai", "anthropic"], required=True)
        sp.add_argument("--model", required=True)
        sp.add_argument("--persona", default="neutral")
        sp.add_argument("--max-output-tokens", type=int, default=8192,
                        help="output cap (the paper's ML runs used 8192); billed only for tokens generated")
        sp.add_argument("--reasoning", default="none",
                        help="OpenAI reasoning_effort: none|minimal|low|medium|high|default(omit); "
                             "omitted automatically for gpt-4o-class models")
        sp.add_argument("--temperature", type=float, default=0.0,
                        help="OpenAI temperature; pass -1 to omit (Anthropic has no temperature)")
        sp.add_argument("--json-mode", choices=["schema", "object", "none"], default="schema",
                        help="OpenAI response_format (schema = json_schema non-strict)")
        sp.add_argument("--thinking-on", action="store_true", help="Anthropic: leave adaptive thinking on")
        sp.add_argument("--no-cache", action="store_true", help="Anthropic: do not mark the shared prefix cacheable")
        sp.add_argument("--cache-ttl", choices=["5m", "1h"], default="1h",
                        help="Anthropic cache TTL (1h recommended for batches; 5m for smoke)")

    s = sub.add_parser("smoke"); gen(s)
    s.add_argument("--student", type=int, default=1); s.add_argument("--q", type=int, default=1)
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("prepare"); gen(s); with_run(s)
    s.add_argument("--students", help="subset, e.g. 1,2,5-10")
    s.add_argument("--limit", type=int, help="first N students only")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("submit"); with_run(s); s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("status"); with_run(s)
    s.add_argument("--wait", action="store_true"); s.add_argument("--interval", type=int, default=60)
    s.add_argument("--timeout", type=int, default=6 * 3600)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("retry"); with_run(s)
    s.add_argument("--max-output-tokens", type=int, default=None, help="override the cap for the retried requests")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_retry)

    s = sub.add_parser("fetch"); with_run(s)
    s.add_argument("--publish", action="store_true",
                   help="also copy the workbook into the exam's results/ and add a tracker row")
    s.set_defaults(fn=cmd_fetch)

    a = p.parse_args()
    if getattr(a, "temperature", None) is not None and a.temperature < 0:
        a.temperature = None
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()

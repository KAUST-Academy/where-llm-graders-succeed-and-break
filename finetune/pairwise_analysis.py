#!/usr/bin/env python3
"""Third-grader (pairwise) comparison: is the AI as good as a human TA?

WHY THIS EXISTS
---------------
The headline comparison used elsewhere in this repo is

    MAE(AI vs TA-average)   vs   inter-grader MAE |TA1 - TA2|

and those two quantities are NOT on the same scale. The AI is scored against the
*average* of two graders -- a less noisy target, because averaging cancels part
of each grader's error -- while the humans are scored against *each other*. The
same adapter scores 2.568 against the TA-average but 2.820 against TA1 alone;
the model didn't change, the target got noisier. So "AI 2.568 < floor 2.61"
compares a 2-grader-averaged error against a 1-vs-1 disagreement and tilts the
result toward the AI.

This script instead treats the AI as a THIRD GRADER and compares like with like:

    |AI - TA1| , |AI - TA2|   vs   |TA1 - TA2|      (all 1-vs-1 disagreements)

and runs a PAIRED test on the per-student difference

    d_i = mean(|AI-TA1|_i, |AI-TA2|_i) - |TA1-TA2|_i

Pairing matters: it removes between-student variance (some papers are just harder
to grade), so the CI is much tighter than comparing two independent means.

Interpretation of the paired CI:
    contains 0        -> AI is statistically indistinguishable from a human TA
    entirely below 0  -> AI agrees with TAs better than they agree with each other
    entirely above 0  -> AI is a worse grader than a TA

Usage:
    python finetune/pairwise_analysis.py                  # all FT runs + D01
    python finetune/pairwise_analysis.py --md out.md      # also write markdown
"""
from __future__ import annotations

import argparse
import glob
import math
import statistics as st
import sys
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import exams  # noqa: E402

RESULTS = HERE / "results"   # finetune results live inside finetune/ now


def exam_for(path: Path) -> "exams.ExamSpec":
    """Cross-exam runs carry the graded exam in the filename (…-on-intro);
    everything else (FT-*, D01) predates the second dataset and is cv."""
    return exams.INTRO if "-on-intro" in path.name else exams.CV


def load_run(path: Path):
    """-> list of (ai_total, ta1_total, ta2_total) for students this run graded.

    TA totals are rebuilt from the per-question columns: the 'TA n - Total Score'
    cells are Excel formulas, and openpyxl returns None for them once a workbook
    has been rewritten (the cached values are dropped).

    Exam conventions differ (and each side must match its own ground truth):
      cv    -> AI Total Score      vs  TA per-question Score (base marks only;
               Q4 has no Score column, so totals are Q1-3)
      intro -> AI Score + AI Bonus vs  TA per-question Grade (bonus folded in)"""
    exam = exam_for(path)
    comb = exam.ta_mode == "combined"
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    col = {n: hdr.index(n) for n in hdr if n}
    if "AI Total Score" not in col:
        return []
    field = "Grade" if comb else "Score"
    qs = [q for q in exam.questions if comb or exam.question_meta[q][0] > 0]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        ai = r[col["AI Total Score"]]
        if not isinstance(ai, (int, float)):
            continue
        ai = float(ai)
        if comb and "AI Total Bonus" in col:
            b = r[col["AI Total Bonus"]]
            if isinstance(b, (int, float)):
                ai += float(b)
        tot = {}
        for ta in (1, 2):
            v = [r[col[f"TA {ta} - Q{q} {field}"]] for q in qs
                 if f"TA {ta} - Q{q} {field}" in col]
            v = [x for x in v if isinstance(x, (int, float))]
            if len(v) == len(qs):
                tot[ta] = sum(v)
        if len(tot) == 2:
            rows.append((ai, float(tot[1]), float(tot[2])))
    return rows


def mean_ci(x):
    if len(x) < 2:
        return (float("nan"),) * 3
    m = st.fmean(x)
    se = st.stdev(x) / math.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


def analyse(rows):
    ai_t1 = [abs(a - t1) for a, t1, _ in rows]
    ai_t2 = [abs(a - t2) for a, _, t2 in rows]
    t1_t2 = [abs(t1 - t2) for _, t1, t2 in rows]
    ai_avg = [abs(a - (t1 + t2) / 2) for a, t1, t2 in rows]
    # Paired: the AI's average 1-vs-1 disagreement minus the humans' 1-vs-1.
    diff = [ (p + q) / 2 - h for p, q, h in zip(ai_t1, ai_t2, t1_t2) ]
    return {
        "n": len(rows),
        "ai_vs_ta_avg": mean_ci(ai_avg),   # the OLD (incomparable) metric
        "ai_vs_ta1": mean_ci(ai_t1),
        "ai_vs_ta2": mean_ci(ai_t2),
        "ta1_vs_ta2": mean_ci(t1_t2),      # the floor, on THESE students
        "paired": mean_ci(diff),
    }


def verdict(paired):
    m, lo, hi = paired
    if math.isnan(m):
        return "n/a"
    if lo <= 0 <= hi:
        return "= TA (n.s.)"
    return "WORSE than TA" if lo > 0 else "BETTER than TA"


EXPLAINER = r"""# Is the fine-tuned grader as good as a human TA?

*This file is generated by `pairwise_analysis.py`. The tables below are the
"third-grader" test. This preamble explains, in plain language, what that test
is and how to read it.*

---

## The question

We want to claim the fine-tuned model *"grades as well as a human TA."* But
humans aren't perfect either: our two TAs disagree with **each other** on many
students. So the real question is not "is the AI perfect?" It is:

> **Does the AI disagree with the TAs about as much as the TAs disagree with
> each other?**

If yes, the AI is just *another grader* -- you could drop it in as a third TA
and not notice. That is the bar.

## The three graders

Every student was graded three times: **TA1**, **TA2**, and the **AI**. For one
student we can measure three disagreements (all out of 35):

```
              AI
            /     \
     |AI-TA1|     |AI-TA2|        how far the AI is from each human
            \     /
   TA1 --- |TA1-TA2| --- TA2      how far the two humans are from each other
```

- `|TA1 - TA2|` = **how much the humans disagree** -- the *human floor*.
- `|AI - TA1|`, `|AI - TA2|` = how much the AI disagrees with each human.

## Comparing like with like

An earlier metric scored the AI against the **average** of the two TAs,
`|AI - (TA1+TA2)/2|`. That is unfair: averaging two graders cancels their noise
and makes an easier target -- one the humans never get (TA1 is judged against
raw TA2, not an average). So we judge the AI the *same way* the humans are
judged, one-on-one, by taking the AI's **average one-on-one disagreement**:

```
AI's disagreement  = ( |AI-TA1| + |AI-TA2| ) / 2
human disagreement = |TA1-TA2|
```

Both are now "how far apart are two graders," measured identically. (The old
`|AI-TA_avg|` column is still shown, but only to connect to earlier numbers --
it is **not** comparable to `|TA1-TA2|`.)

## The paired difference -- and why it is "per student"

For **each student** we subtract, using *that same student's* numbers:

```
d = (AI's avg one-on-one disagreement)  -  (the humans' disagreement)
```

Why per student? Because some exams are just ambiguous -- *everyone* disagrees on
them. Example:

| student | AI's disagreement | humans' disagreement | d |
|---|---|---|---|
| Ali  | 3.0 | 2.5 | +0.5 |
| Sara | 9.0 | 8.5 | +0.5 |
| Omar | 1.0 | 0.5 | +0.5 |

Sara looks alarming (AI off by 9!) until you see the humans were 8.5 apart on
her too -- she is simply a confusing exam, not an AI failure. Subtracting
*within* each student removes that per-student difficulty, leaving only the
thing we care about: **was the AI a bigger outlier than a human would be, on
this exam?** (Averaging everything first and subtracting at the end would let
exam difficulty leak into the comparison; pairing is immune to that.)

Reading a single `d`:
- **d < 0** -> the AI agreed with the humans *more* than they agreed with each
  other (AI looks like a good grader).
- **d > 0** -> the AI was the odd one out.
- **d ~ 0** -> the AI fit right in.

## The confidence interval -- why "[lo, hi]" is there

We measured 114 students, but that is a **sample**. A different 114 students
would give a slightly different average `d`. So the single average is an
*estimate* with wobble. The 95% confidence interval `[lo, hi]` is the believable
range for the *true* average (computed as `mean +/- 1.96 x standard error`).

The verdict is decided by **where that range sits relative to 0**:

| 95% range | meaning | verdict |
|---|---|---|
| straddles 0 (lo <= 0 <= hi) | cannot tell the AI apart from a human TA | **= TA** |
| entirely **below** 0 | AI agrees with humans *more* than they agree with each other | **BETTER than TA** |
| entirely **above** 0 | AI is a bigger outlier than a human | **WORSE than TA** |

The interval is what stops us from over-claiming on a lucky sample: an average
`d` of -0.3 only means "better than a TA" if the *whole* range is below 0. If the
honest range is `[-0.7, +0.1]`, that -0.3 could just be noise from which 114
students we happened to grade -> we can only say **"= TA"** (indistinguishable),
which is the safe, honest claim. "95%" means this recipe traps the true value 95
times out of 100.

---

"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", type=Path, default=None, help="also write a markdown table")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run tags (default: all FT-* plus D01)")
    args = ap.parse_args()

    if args.runs:
        paths = [RESULTS / f"{t}.xlsx" for t in args.runs]
    else:
        paths = sorted(RESULTS.glob("FT-*.xlsx"))
        d01 = sorted(RESULTS.glob("D01__*.xlsx"))
        paths += d01

    out = []
    for p in paths:
        if not p.exists():
            print(f"  !! missing: {p.name}", file=sys.stderr)
            continue
        rows = load_run(p)
        if not rows:
            continue
        tag = p.name.split("__")[0].replace(".xlsx", "")
        out.append((tag, analyse(rows)))

    hdr = (f"{'run':24s} {'n':>4} {'|AI-TAavg|':>10} {'|AI-TA1|':>9} "
           f"{'|TA1-TA2|':>10} {'paired diff':>12}  verdict")
    print(hdr); print("-" * len(hdr))
    lines = []
    for tag, s in out:
        pm, plo, phi = s["paired"]
        line = (f"{tag:24s} {s['n']:>4} {s['ai_vs_ta_avg'][0]:10.3f} "
                f"{s['ai_vs_ta1'][0]:9.3f} {s['ta1_vs_ta2'][0]:10.3f} "
                f"{pm:+7.3f} [{plo:+.2f},{phi:+.2f}]".ljust(12) +
                f"  {verdict(s['paired'])}")
        print(line)
        lines.append((tag, s))

    print("\nNotes:")
    print("  |AI-TAavg| is the OLD metric -- vs a 2-grader average, NOT comparable")
    print("            to |TA1-TA2|. Shown only to connect to earlier numbers.")
    print("  |AI-TA1| vs |TA1-TA2| is the like-for-like comparison.")
    print("  paired diff = mean(|AI-TA1|,|AI-TA2|) - |TA1-TA2|, per student.")
    print("            CI contains 0 -> AI indistinguishable from a human TA.")

    if args.md:
        with open(args.md, "w") as f:
            f.write(EXPLAINER)
            f.write("| run | n | \\|AI−TA_avg\\| | \\|AI−TA1\\| | \\|AI−TA2\\| | "
                    "\\|TA1−TA2\\| | paired diff [95% CI] | verdict |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
            for tag, s in lines:
                pm, plo, phi = s["paired"]
                f.write(f"| {tag} | {s['n']} | {s['ai_vs_ta_avg'][0]:.3f} | "
                        f"{s['ai_vs_ta1'][0]:.3f} | {s['ai_vs_ta2'][0]:.3f} | "
                        f"{s['ta1_vs_ta2'][0]:.3f} | {pm:+.3f} "
                        f"[{plo:+.2f}, {phi:+.2f}] | {verdict(s['paired'])} |\n")
        print(f"\nWrote {args.md}")


if __name__ == "__main__":
    main()

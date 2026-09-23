"""
Auto-grader ablation analysis — single entry point.

Produces:
  analysis/computer_vision_master_comparison.csv
  analysis/computer_vision_master_comparison.md
  analysis/computer_vision_human_floor.md
  analysis/computer_vision_variance.md
  analysis/computer_vision_failure_cases.md
  media/*.pdf

Per-question MAE is reported against the two-TA per-question average
(`TA {1,2} - Q{1,2,3} Score`, present for all 570 students), plus per-question
mean/std of AI scores. Q4 is bonus-only and has no TA Score column.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE = Path(__file__).resolve().parent.parent
ANALYSIS = BASE / "analysis"
FIGURES = BASE / "media"
ANALYSIS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Configuration: ablation metadata (extracted from filename + handoff)
# -----------------------------------------------------------------------------

RESULTS_DIR = BASE / "computer_vision_results" / "results"
VARIANCE_DIR = BASE / "computer_vision_results" / "variance"


def _source_from_rid(rid: str) -> str:
    """Locally-served run_ids are prefixed with 'L-'; the cross-vendor closed runs
    of Section 6.3 use O (OpenAI, Batch API) and N (Anthropic); everything else
    is Gemini via Vertex. Anything not 'local' is a closed model."""
    if rid.startswith("L-"):
        return "local"
    if rid.startswith("O"):
        return "openai"
    if rid.startswith("N"):
        return "anthropic"
    return "gemini"


# Model-family taxonomy. The open-weights side stopped being Qwen-only in July
# 2026 — it now spans seven families — but the summaries, the cost map and the
# figure specs each hardcoded "qwen" separately. That mislabels the 81 non-Qwen
# local runs in prose and silently drops them from figures, and it makes the
# collapse look like a Qwen property when the whole point of the new runs is to
# test whether it generalises. Add a family here and every consumer follows.
# Runs below this are deliberate small-n designs (the students-1-100 Gemini
# subsets, the n~50 temperature-variance repeats), not full-cohort ablations.
# Their MAE is not comparable to a 570-student run — the human floor on the
# 1-100 cohort is 3.43, not 2.61 — so aggregate claims must exclude them.
# Previously spelled as a bare `500` in three places.
FULL_COHORT_MIN_N = 500

# --- refusal vs collapse -----------------------------------------------------
# MAE alone cannot separate "graded everything at zero" from "graded harshly but
# still ranked students": both saturate near 26.04, which is just the distance
# from the TA mean and therefore a ceiling artefact, not a severity measurement.
# The measured data separate cleanly on zero-rate and on AI-vs-TA correlation:
#
#   refusal       zero-rate  99-100%   stdev 0.00-0.09   r  undefined / -0.01
#   near-refusal  zero-rate  91-99%    stdev 0.63-3.33   r  0.07 - 0.27
#   collapse      zero-rate  13-47%    stdev 4.51-10.14  r  0.47 - 0.78
#   graded        zero-rate   0-0.4%   stdev 4.97-7.08   r  0.76 - 0.95
#
# There is a wide empty band between near-refusal and collapse on every one of
# the three axes, so the thresholds below are not knife-edge.
REFUSAL_ZERO_RATE = 0.90     # awarded nothing to >=90% of students
REFUSAL_STDEV = 0.5          # ...and did not vary at all while doing it
COLLAPSE_MAE = 8.0           # ~3x the 2.61 human inter-grader floor


def behaviour_class(zero_rate: float, mae: float, stdev: float) -> str:
    """Classify a run as refusal / near-refusal / collapse / graded.

    A refusal is not a severe collapse — it is a different failure, and
    reporting the two under one heading conflates "grades harshly" with
    "does not grade". Keep them apart in every table that quotes MAE.
    """
    if pd.isna(mae):
        return "unmeasured"
    if not pd.isna(zero_rate) and zero_rate >= REFUSAL_ZERO_RATE:
        return "refusal" if (not pd.isna(stdev) and stdev < REFUSAL_STDEV) \
            else "near-refusal"
    return "collapse" if mae >= COLLAPSE_MAE else "graded"

OPEN_WEIGHT_FAMILIES = (
    ("qwen", "Qwen"),
    ("llama", "Llama"),
    ("glm", "GLM"),
    ("gemma", "Gemma"),
    ("mistral", "Mistral"),
    ("deepseek", "DeepSeek"),
)
# The cross-vendor closed runs (GPT-5.5, GPT-5.4, Claude Opus 5) sit in the
# master CSV beside the Gemini ones. `family != "Gemini"` meant open weights
# only until they arrived, and then counted 20 open pairs instead of 17. The
# paper reports them in tab:closedvendors, not in the strict-ladder tables.
CROSS_VENDOR_FAMILIES = ("OpenAI", "Anthropic")
CLOSED_FAMILIES = ("Gemini",) + CROSS_VENDOR_FAMILIES


def model_family(model: str, source: str | None = None) -> str:
    """Family label for a parsed model string, e.g. 'llama-3.3-70b' -> 'Llama'.

    Matched on prefix so a new size/variant needs no change here. Anything not
    an open-weight family is Gemini when the run-id says so; 'other' otherwise,
    which is deliberately visible rather than silently folded into a family.
    """
    m = str(model).lower()
    for prefix, family in OPEN_WEIGHT_FAMILIES:
        if m.startswith(prefix):
            return family
    if source == "openai" or m.startswith("gpt"):
        return "OpenAI"
    if source == "anthropic" or m.startswith("claude"):
        return "Anthropic"
    if source == "gemini" or "gemini" in m or "flash" in m or "pro" in m:
        return "Gemini"
    return "other"


def family_breakdown(df: pd.DataFrame) -> str:
    """'56 Qwen + 24 Gemini + 14 Mistral + ...', ordered by count."""
    counts = df.apply(
        lambda r: model_family(r["model"], r.get("source")), axis=1
    ).value_counts()
    return " + ".join(f"{n} {fam}" for fam, n in counts.items())


# --- parameter counts --------------------------------------------------------
# Read off the model name wherever the vendor put it there (`llama-3.3-70b` -> 70).
# Three models in this study do not carry it. None of the three is guessed -- each
# value below has a source, and the sources are not equally strong:
#
#   deepseek-coder-v2-lite   16B total / 2.4B active   MEASURED from the safetensors
#                                                      index (31.4 GB bf16).
#   glm-4.5-air             106B total /  12B active   MEASURED, but note the index
#                                                      `total_size` under-reports by
#                                                      exactly 2x here; real files
#                                                      are ~221 GB.
#   qwen3-coder-next         80B MoE  /   3B active    RECORDED, not measured: the
#                                                      checkpoint is not cached
#                                                      locally, so this comes from
#                                                      computer_vision_grade_with_local.py:101, and
#                                                      the paper already prints 80B
#                                                      (sections/brittleness.tex:34).
#
# Anything not listed and not carrying a size in its name returns None and drops
# out of the size-ordered ladder rather than being assigned an invented number,
# which would silently reorder the very thing the ladder is read for.
MODEL_PARAMS_B: dict[str, float] = {
    "deepseek-coder-v2-lite": 16,
    "glm-4.5-air": 106,
    "qwen3-coder-next": 80,
}


def model_params_b(model: str) -> float | None:
    """Total parameters in billions, or None when it is not verifiable.

    MoE models report TOTAL, not active: the ladder is about model capacity, and
    mixing the two would file qwen3-coder-30b-a3b (30B total, 3B active) three
    rungs below where its capacity actually sits.
    """
    m = str(model).lower()
    if m in MODEL_PARAMS_B:
        return float(MODEL_PARAMS_B[m])
    # The FIRST `<digits>b` token wins: qwen3-235b-a22b is a 235B model with 22B
    # active, and the trailing `a22b` must not take the match.
    hit = re.search(r"(\d+(?:\.\d+)?)b\b", m)
    return float(hit.group(1)) if hit else None


# The prompt configuration every persona sweep was run at. Pinning all six fields
# is what makes a strict/neutral difference attributable to the persona sentence.
DEFAULT_PROMPT_CONFIG = dict(use_solution=1, use_guidelines=1, use_reasoning=0,
                             use_breakdown=1, temperature=0.0, few_shot=0)


def matched_persona_pairs(master_df: pd.DataFrame, persona: str = "strict",
                          baseline: str = "neutral") -> pd.DataFrame:
    """One row per model: `persona` vs `baseline` at an IDENTICAL prompt config.

    Taking each model's *best* baseline instead -- the obvious shortcut -- pairs
    sol0 with sol1, bd0 with bd1, and few-shot with zero-shot. That is not a
    hypothetical: it produced four wrong ratios in an earlier draft of this
    analysis, including x6.50 for qwen2.5-coder-32b whose matched value is x2.71.
    """
    sel = master_df[master_df["n_valid_vs_TA"] >= FULL_COHORT_MIN_N].copy()
    for k, v in DEFAULT_PROMPT_CONFIG.items():
        if k in sel.columns:
            sel = sel[sel[k] == v]

    def pick(g: pd.DataFrame) -> pd.Series:
        # Prefer a single-run row over an averaged variance-repeat set (A01 over
        # E01 for flash-lite), then the larger cohort, then run_id for stability.
        g = g.assign(_multi=(g["runs"].fillna(1) != 1).astype(int))
        return g.sort_values(["_multi", "n_valid_vs_TA", "run_id"],
                             ascending=[True, False, True]).iloc[0]

    rows = []
    for model, g in sel.groupby("model"):
        base, test = g[g["strictness"] == baseline], g[g["strictness"] == persona]
        if not len(base) or not len(test):
            continue
        b_r, t_r = pick(base), pick(test)
        rows.append({
            "family": model_family(model, b_r.get("source")),
            "model": model,
            "params_b": model_params_b(model),
            "baseline_rid": b_r["run_id"],
            "test_rid": t_r["run_id"],
            "baseline_MAE": float(b_r["MAE_vs_TA_avg"]),
            "test_MAE": float(t_r["MAE_vs_TA_avg"]),
            "ratio": float(t_r["MAE_vs_TA_avg"]) / float(b_r["MAE_vs_TA_avg"]),
            "test_mean": float(t_r["mean_total_score"]),
            "test_stdev": float(t_r["std_total_score"]),
            "test_behaviour": t_r["behaviour"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)\
                             .reset_index(drop=True)


# --- how a strict run fails --------------------------------------------------
# Three mechanisms, separated on the Q1 evidence. Both thresholds sit in a wide
# empty band of the measured data (17 matched strict runs):
#
#   Q1-zero rate              0.0 - 5.4%  |  24 - 100%     -> gap, cut at 10%
#   of those, Q2 > 0          0   -  15%  |  51 -  77%     -> gap, cut at 35%
#     (evaluated only where the Q1-zero rate clears 10%)
#
# The distinction matters because all three land in the same MAE range while
# being different failures: "grades everything harshly" is not "zeroes one field"
# is not "awards nothing at all".
MECH_ZERO_RATE = 0.10
MECH_SELECTIVE = 0.35


def mechanism_class(mc: dict | None) -> str:
    """uniform severity / selective field collapse / blanket zeroing."""
    if not mc:
        return "unmeasured"
    if mc["frac_zero"] < MECH_ZERO_RATE:
        return "uniform severity"
    return ("selective field collapse" if mc["frac_selective"] >= MECH_SELECTIVE
            else "blanket zeroing")


VAR_BY_RID: dict = {}


def parse_run_tag(stem: str) -> dict:
    """Parse '<RID>__<RID>_m-<model>_sol0_gd1_rs0_bd1_str-neutral_t00_n1' → dict."""
    # Take after the first '__' if present
    if "__" in stem:
        rid, rest = stem.split("__", 1)
    else:
        rid, rest = stem.split("_", 1)
    # rest now starts with "<RID>_m-<model>_..."
    parts = rest.split("_")
    # Find model token
    m_idx = next(i for i, p in enumerate(parts) if p.startswith("m-"))
    model = parts[m_idx][2:]
    # Some models contain dots/dashes — gather until next sol/gd/rs/bd/str/t/n token
    j = m_idx + 1
    while j < len(parts) and not re.match(r"^(sol|gd|rs|bd|str|t|n)\d|^str-", parts[j]):
        model += "_" + parts[j]
        j += 1
    # few_shot defaults to 0 and is overridden by a trailing `_fs<N>`. It MUST be
    # carried into the master table: without it L-K01 (`_fs2`, tracker few_shot=2)
    # is indistinguishable from a zero-shot run, and any config-matched comparison
    # silently pairs it with zero-shot runs. That is not hypothetical — it produced
    # a wrong strict/neutral ratio for qwen3-coder-30b-a3b in an analysis written
    # against this CSV.
    cfg = {"run_id": rid, "model": model, "few_shot": 0}
    for p in parts[j:]:
        if p.startswith("sol"):
            cfg["use_solution"] = int(p[3:])
        elif p.startswith("gd"):
            cfg["use_guidelines"] = int(p[2:])
        elif p.startswith("rs"):
            cfg["use_reasoning"] = int(p[2:])
        elif p.startswith("bd"):
            cfg["use_breakdown"] = int(p[2:])
        elif p.startswith("str-"):
            cfg["strictness"] = p[4:]
        elif p.startswith("t") and p[1:].isdigit():
            cfg["temperature"] = int(p[1:]) / 10.0
        elif p == "tvd":
            # Anthropic exposes no sampling temperature on current models: the
            # closed_batch_grade.py runs carry 'tvd' (vendor default). Recorded as
            # 0 so the default-configuration filters keep the run; the appendix
            # table prints it as '---'.
            cfg["temperature"] = 0.0
        elif p.startswith("n") and p[1:].isdigit():
            cfg["runs"] = int(p[1:])
        elif p.startswith("fs") and p[2:].isdigit():
            cfg["few_shot"] = int(p[2:])
    return cfg


def list_ablations():
    """Discover every result xlsx across the four result dirs.

    Each ablation is uniquely identified by `run_id` (parsed from the filename
    stem). If the same run_id appears in multiple dirs (it shouldn't, but
    might during transitions), the FIRST one wins so the original 30 ablations
    take precedence over any duplicate in the new dirs.
    """
    rows = []
    seen_rids: set[str] = set()
    if not RESULTS_DIR.is_dir():
        return rows
    for f in sorted(RESULTS_DIR.glob("*.xlsx")):
        cfg = parse_run_tag(f.stem)
        rid = cfg["run_id"]
        if rid in seen_rids:
            print(f"   (skipping duplicate of {rid} found at {f})")
            continue
        seen_rids.add(rid)
        cfg["source"] = _source_from_rid(rid)
        cfg["xlsx"] = f
        for vf in VARIANCE_DIR.glob(f"{rid}_*.csv"):
            VAR_BY_RID[rid] = vf
            break
        rows.append(cfg)
    return rows


# -----------------------------------------------------------------------------
# 2. Master comparison table
# -----------------------------------------------------------------------------

SUBMISSIONS = BASE / "computer_vision_dataset" / "submissions_extracted"


def has_submission(n: int) -> bool:
    return (SUBMISSIONS / str(n)).is_dir()


def load_result(f: Path) -> pd.DataFrame:
    df = pd.read_excel(f, sheet_name=0)
    return df


def _ta_grader_total(df: pd.DataFrame, grader: int) -> pd.Series:
    """One TA's base total, rebuilt from their per-question scores.

    Q4 has no `Score` column — it is bonus-only — so the 35-pt base total is
    Q1+Q2+Q3 by construction, which is exactly what the workbook's
    `TA {n} - Total Score (out of 35)` formula computes.

    skipna=False on purpose: a grader missing a question is *missing*, not a
    partial sum. Summing with skipna=True would turn an absent grader into a
    real 0 and drag the two-TA average down by half their score.
    """
    cols = [f"TA {grader} - Q{q} Score" for q in (1, 2, 3)]
    stored_col = f"TA {grader} - Total Score (out of 35)"
    if not all(c in df.columns for c in cols):
        # Defensive: a workbook shaped differently than any we have seen.
        if stored_col in df.columns:
            return pd.to_numeric(df[stored_col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)
    per_q = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cols], axis=1)
    return per_q.sum(axis=1, skipna=False)


def ta_per_question(df: pd.DataFrame, q: int) -> pd.Series:
    """Two-TA average score on question `q` — the real per-question ground truth.

    Only Q1–Q3 exist: Q4 is bonus-only and has no `Score` column, which is why
    the 35-pt base scale is Q1+Q2+Q3.

    This replaces the cross-run median that per-question MAE used to be measured
    against. That median was computed over every discovered run with no persona
    filter, so roughly a third of its inputs were strict-collapse runs pulling it
    toward zero — and, worse, it MOVED whenever unrelated runs were added.
    Growing the run set from 69 to 161 shifted published per-question MAEs by up
    to 0.7 pts (A01 Q1 0.84 -> 1.50) with no change to any underlying grade.
    A reference that drifts is not a reference.
    """
    cols = [f"TA {n} - Q{q} Score" for n in (1, 2)]
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    vals = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in present], axis=1)
    return vals.mean(axis=1, skipna=True)


def _check_rebuild_against_stored(df: pd.DataFrame, t1: pd.Series,
                                  t2: pd.Series) -> None:
    """Regression guard: where the stored totals survived, the rebuild must match.

    69 of the 161 workbooks still carry literal TA totals. On those the rebuilt
    value reproduces the stored one to ~3.6e-15, so any real disagreement means
    the per-question columns and the total column have drifted apart and every
    number downstream is suspect.
    """
    for grader, rebuilt in ((1, t1), (2, t2)):
        col = f"TA {grader} - Total Score (out of 35)"
        if col not in df.columns:
            continue
        stored = pd.to_numeric(df[col], errors="coerce")
        overlap = stored.notna() & rebuilt.notna()
        if not overlap.any():
            continue
        worst = float((rebuilt[overlap] - stored[overlap]).abs().max())
        if worst > 1e-6:
            warnings.warn(
                f"TA {grader}: rebuilt base total disagrees with the stored "
                f"'{col}' by up to {worst:.4g} over {int(overlap.sum())} "
                f"students. The per-question columns and the total column have "
                f"diverged; MAE/bias for this run cannot be trusted.",
                RuntimeWarning, stacklevel=3,
            )


def ta_combined(df: pd.DataFrame) -> pd.Series:
    """Effective TA total score per student: avg of available TAs.

    Rebuilt from the per-question `TA {n} - Q{1,2,3} Score` columns rather than
    read from `TA {n} - Total Score (out of 35)`. Those total columns hold Excel
    FORMULAS, and the grader's clone path (shutil.copy2 -> openpyxl
    load_workbook(data_only=False) -> save) drops their cached values, so they
    are empty in every workbook written after cc66e2e — 92 of the 161 result
    files, i.e. every new-family run. Reading them produced MAE/bias/CI = NaN
    with no error raised, and the NaNs then propagated into the generated
    summaries as blank cells and miscounted totals.

    The per-question columns are literal data and complete (570/570 in all 161
    workbooks), so the rebuild is exact — see _check_rebuild_against_stored.
    """
    t1 = _ta_grader_total(df, 1)
    t2 = _ta_grader_total(df, 2)
    _check_rebuild_against_stored(df, t1, t2)
    # Treat a TA score as missing iff NaN. Keep 0s as real zeros.
    both = (t1.notna()) & (t2.notna())
    only1 = t1.notna() & t2.isna()
    only2 = t2.notna() & t1.isna()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[both] = (t1[both] + t2[both]) / 2.0
    out[only1] = t1[only1]
    out[only2] = t2[only2]
    return out


def maybe_fill_from_variance(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """For E rows the result xlsx is empty; reconstruct per-question score
    as the mean across runs of the variance CSV.

    Two repairs vs the naive reconstruction (which summed Q1-Q4 `score`):
      * The E-series variance CSVs predate the Q4-as-bonus fix, so the Q4
        grade sits in the `score` column. Route it into `AI Q4 Bonus` and
        keep `AI Q4 Score` at 0, matching the Q4-as-bonus repair already
        applied to the released result spreadsheets.
      * `AI Total Score` sums Q1-Q3 base only — the 35-pt scale the TAs and
        every other run use. Summing Q4 in both inflated the total (out of
        40) and, via skipna=False, silently restricted the cohort to the
        ~174/570 students who attempted the bonus question.
    """
    if df["AI Total Score"].notna().any():
        return df
    var_path = VAR_BY_RID.get(run_id)
    if var_path is None or not var_path.exists():
        return df
    v = pd.read_csv(var_path)
    means = v.groupby(["student", "question"])["score"].mean().unstack("question")
    bmeans = v.groupby(["student", "question"])["bonus"].mean().unstack("question")
    df = df.copy()
    for q in [1, 2, 3]:
        if q in means.columns:
            df.loc[means.index, f"AI Q{q} Score"] = means[q].values
        if q in [1, 2] and q in bmeans.columns:
            df.loc[bmeans.index, f"AI Q{q} Bonus"] = bmeans[q].values
    if 4 in means.columns:
        q4 = means[4].fillna(0)
        if 4 in bmeans.columns:
            q4 = q4 + bmeans[4].fillna(0)
        df.loc[means.index, "AI Q4 Bonus"] = q4.values
        df.loc[means.index, "AI Q4 Score"] = 0.0
    q_cols = [f"AI Q{q} Score" for q in [1, 2, 3]]
    df["AI Total Score"] = df[q_cols].sum(axis=1, skipna=False)
    return df


def bootstrap_mae_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for MAE = mean(|diffs|)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.abs(diffs[idx]).mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def build_master_table(ablations):
    per_q_panels = {q: [] for q in [1, 2, 3, 4]}
    per_q_cols = {1: "AI Q1 Score", 2: "AI Q2 Score", 3: "AI Q3 Score", 4: "AI Q4 Score"}
    cached = {}
    for a in ablations:
        df = load_result(a["xlsx"]).set_index("Number")
        df = maybe_fill_from_variance(df, a["run_id"])
        cached[a["run_id"]] = df
        for q, col in per_q_cols.items():
            s = pd.to_numeric(df[col], errors="coerce")
            s.name = a["run_id"]
            per_q_panels[q].append(s)

    rows = []
    for a in ablations:
        df = cached[a["run_id"]]
        ta_avg = ta_combined(df.reset_index()).values
        ta_avg = pd.Series(ta_avg, index=df.index)
        ai_total = pd.to_numeric(df["AI Total Score"], errors="coerce")
        # The grader writes AI Total Score by summing Q1-Q4, so a failed Q4 --
        # the bonus-only question this paper excludes from the 35-point scale --
        # nulls the total even though the base score is fully present. Recover
        # it from Q1-Q3, which IS the base scale. Same reasoning as
        # maybe_fill_from_variance(); it just never applied outside the E-series.
        _base = pd.concat([pd.to_numeric(df[f"AI Q{q} Score"], errors="coerce")
                           for q in (1, 2, 3)], axis=1)
        _recoverable = ai_total.isna() & _base.notna().all(axis=1)
        if _recoverable.any():
            ai_total = ai_total.where(~_recoverable, _base.sum(axis=1))
        # graded == AI Total Score non-null AND not 0 OR has-submission
        graded_mask = ai_total.notna()
        # diff vs TA
        valid = graded_mask & ta_avg.notna()
        diff = ai_total[valid] - ta_avg[valid]
        mae = diff.abs().mean()
        bias = diff.mean()
        mae_lo, mae_hi = bootstrap_mae_ci(diff.values)

        # --- discrimination: does this grader tell students apart at all? ---
        # MAE cannot distinguish a REFUSAL (every student zeroed; MAE saturates
        # at the distance to the TA mean, ~26.04, which is a ceiling artefact)
        # from a COLLAPSE (harsh but still ranking students). Llama-3.1-8B+strict
        # and Qwen2.5-Coder-32B+strict differ by 5.75 MAE but by 4.52 stdev and
        # 100 vs 0 percentage points of zero-rate. Emit both so the distinction
        # is a column rather than a hand judgement.
        zero_rate = (
            float((ai_total[graded_mask] == 0).mean()) if graded_mask.any()
            else float("nan")
        )
        if valid.sum() >= 3 and ai_total[valid].std() > 0 and ta_avg[valid].std() > 0:
            ai_ta_r = float(_scipy_stats.pearsonr(ai_total[valid], ta_avg[valid])[0])
        else:
            # Undefined, not zero: a constant grader has no correlation to report.
            ai_ta_r = float("nan")

        # n_failed: AI Total == 0 AND has submission
        zero_score = (ai_total == 0)
        submitted = pd.Series([has_submission(n) for n in df.index], index=df.index)
        n_failed = int((zero_score & submitted).sum())
        # Per-question MAE against the two-TA per-question average (Q1-Q3).
        # Q4 is bonus-only and has no TA Score column, so it gets a mean but no
        # MAE — there is nothing to compare it to on the 35-pt base scale.
        per_q_mae = {}
        per_q_mean = {}
        per_q_std = {}
        for q, col in per_q_cols.items():
            s = pd.to_numeric(df[col], errors="coerce")
            per_q_mean[q] = float(s.mean())
            per_q_std[q] = float(s.std())
            if q == 4:
                continue
            ref = ta_per_question(df, q)
            v = s.notna() & ref.notna()
            per_q_mae[q] = float((s[v] - ref[v]).abs().mean())

        rows.append({
            "run_id": a["run_id"],
            "source": a["source"],
            "model": a["model"],
            "use_solution": a.get("use_solution"),
            "use_guidelines": a.get("use_guidelines"),
            "use_reasoning": a.get("use_reasoning"),
            "use_breakdown": a.get("use_breakdown"),
            "strictness": a.get("strictness"),
            "temperature": a.get("temperature"),
            "runs": a.get("runs"),
            "few_shot": a.get("few_shot", 0),
            "n_graded": int(graded_mask.sum()),
            "n_valid_vs_TA": int(valid.sum()),
            "mean_total_score": float(ai_total[graded_mask].mean()),
            "std_total_score": float(ai_total[graded_mask].std()),
            "MAE_vs_TA_avg": float(mae),
            "MAE_CI95_lo": mae_lo,
            "MAE_CI95_hi": mae_hi,
            "bias_vs_TA_avg": float(bias),
            "zero_rate": zero_rate,
            "AI_TA_pearson_r": ai_ta_r,
            "behaviour": behaviour_class(zero_rate, float(mae),
                                         float(ai_total[graded_mask].std())),
            "n_failed": n_failed,
            "Q1_MAE_vs_TA": per_q_mae[1],
            "Q2_MAE_vs_TA": per_q_mae[2],
            "Q3_MAE_vs_TA": per_q_mae[3],
            "Q1_mean": per_q_mean[1],
            "Q2_mean": per_q_mean[2],
            "Q3_mean": per_q_mean[3],
            "Q4_mean": per_q_mean[4],
        })
    return pd.DataFrame(rows), cached


# -----------------------------------------------------------------------------
# 3. Human floor (TA1 vs TA2)
# -----------------------------------------------------------------------------

def human_floor():
    """Inter-grader disagreement.

    Structure: 20 TAs total, organized in 10 fixed pairs (slot-1 TA always
    paired with the same slot-2 TA). Each pair grades ~55–60 students. So
    'TA_1_ID'/'TA_2_ID' are *slots*, not identities; their slot-wise diff
    cannot be interpreted as a systematic bias. We report:
      - pooled inter-grader MAE / Pearson r across all pairs (the headline floor)
      - per-pair MAE / Pearson r (10 numbers — heterogeneity of the floor)
      - per-TA mean score given (harshness, 20 numbers)
    """
    df = pd.read_excel(BASE / "computer_vision_dataset/Practical_AI_exam_grades.xlsx")
    t1 = pd.to_numeric(df["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(df["TA 2 - Total Score (out of 35)"], errors="coerce")
    b1 = pd.to_numeric(df["TA 1 - Total Bonus (out of 13)"], errors="coerce")
    b2 = pd.to_numeric(df["TA 2 - Total Bonus (out of 13)"], errors="coerce")

    both = t1.notna() & t2.notna()  # a 0/0 grade is a real dual grade
    n = int(both.sum())
    signed_diff = (t1[both] - t2[both])  # for bootstrap
    diff = signed_diff.abs()
    mae = float(diff.mean())
    mae_lo, mae_hi = bootstrap_mae_ci(signed_diff.values)
    pearson = float(np.corrcoef(t1[both], t2[both])[0, 1])
    max_dis = float(diff.max())
    bins = [0, 1, 2, 3, 5, 8, 11, 35]
    counts = pd.cut(diff, bins=bins, right=False).value_counts().sort_index()

    both_b = b1.notna() & b2.notna() & ~((b1 == 0) & (b2 == 0))
    bonus_mae = float((b1[both_b] - b2[both_b]).abs().mean()) if both_b.sum() else float("nan")

    # Per-pair stats
    sub = df[both].copy()
    sub["pair"] = sub.apply(lambda r: tuple(sorted([r["TA_1_ID"], r["TA_2_ID"]])), axis=1)
    per_pair = []
    for pair, g in sub.groupby("pair"):
        a = pd.to_numeric(g["TA 1 - Total Score (out of 35)"], errors="coerce")
        b = pd.to_numeric(g["TA 2 - Total Score (out of 35)"], errors="coerce")
        per_pair.append({
            "pair": " / ".join(pair),
            "n": len(g),
            "MAE": float((a - b).abs().mean()),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(g) > 1 else float("nan"),
            "mean_diff": float((a - b).mean()),  # signed: positive = first listed grades higher
        })
    per_pair_df = pd.DataFrame(per_pair).sort_values("MAE")

    # Per-TA "harshness" — mean score that TA awarded across all submissions they graded
    long = pd.concat([
        pd.DataFrame({"TA": df["TA_1_ID"], "score": t1}),
        pd.DataFrame({"TA": df["TA_2_ID"], "score": t2}),
    ]).dropna()
    per_ta = long.groupby("TA")["score"].agg(["count", "mean", "std"]).round(2).sort_values("mean")

    return {
        "n_pairs": n,
        "MAE": mae,
        "MAE_CI95_lo": mae_lo,
        "MAE_CI95_hi": mae_hi,
        "pearson": pearson,
        "max_disagreement": max_dis,
        "bins": counts,
        "bonus_MAE": bonus_mae,
        "bonus_n": int(both_b.sum()),
        "per_pair": per_pair_df,
        "per_ta": per_ta,
        "n_tas": int(long["TA"].nunique()),
    }


# -----------------------------------------------------------------------------
# 4. Variance (E rows)
# -----------------------------------------------------------------------------

def variance_analysis():
    out = []
    var_files = [
        ("gemini-E01-t0.0", VARIANCE_DIR / "E01_m-flash-lite_sol1_gd1_rs0_bd1_str-neutral_t00_n5.csv"),
        ("gemini-E02-t0.5", VARIANCE_DIR / "E02_m-flash-lite_sol1_gd1_rs0_bd1_str-neutral_t05_n5.csv"),
        ("gemini-E03-t0.7", VARIANCE_DIR / "E03_m-flash-lite_sol1_gd1_rs0_bd1_str-neutral_t07_n5.csv"),
        ("local-E01-qwen32b-t0.5", VARIANCE_DIR / "L-E01_m-qwen2.5-coder-32b_sol1_gd1_rs0_bd1_str-neutral_t05_n5.csv"),
        ("local-E02-qwen3coder-next-t0.5", VARIANCE_DIR / "L-E02_m-qwen3-coder-next_sol1_gd1_rs0_bd1_str-neutral_t05_n5.csv"),
    ]
    per_cell_panels = {}
    for label, path in var_files:
        if not path.exists():
            print("MISSING", path); continue
        v = pd.read_csv(path)
        # Q1-Q3 only. Q4 is bonus-only and excluded from the 35-point base scale
        # everywhere else in this paper; leaving it in here added ~180 near-
        # constant cells per configuration (95% of them zero-variance at t=0.5),
        # which dragged the median per-cell sd down and understated the tail.
        v = v[v["question"] != 4]
        # Per (student, question) compute sample stdev across runs (use ddof=1 if n>=2)
        gb = v.groupby(["student", "question"])["score"]
        std = gb.std(ddof=1)
        n_runs = gb.size()
        mean = gb.mean()
        per_cell_panels[label] = pd.DataFrame({"mean": mean, "std": std, "n": n_runs})
        valid = std.dropna()
        out.append({
            "config": label,
            "n_cells": int(len(valid)),
            "mean_runs_per_cell": float(n_runs.mean()),
            "mean_cell_std": float(valid.mean()),
            "median_cell_std": float(valid.median()),
            "max_cell_std": float(valid.max()),
            "pct_cells_zero_std": float((valid == 0).mean() * 100),
            "pct_cells_std_le_0_5": float((valid <= 0.5).mean() * 100),
            "pct_cells_std_gt_1": float((valid > 1).mean() * 100),
        })
    return pd.DataFrame(out), per_cell_panels


# -----------------------------------------------------------------------------
# 5. Failure cases — use best-config result, find |AI_total - TA_avg| > 5
# -----------------------------------------------------------------------------

def failure_cases(master_df: pd.DataFrame, cached: dict,
                  min_n: int = FULL_COHORT_MIN_N):
    """Pick the best full-scale config (n_graded >= min_n) for failure-case
    vignettes. Avoids picking the n=100 Gemini G/M subsets, whose MAE is
    computed on only 1-100 and isn't directly comparable to n=570 runs."""
    full_scale = master_df[master_df["n_graded"] >= min_n]
    pool = full_scale if len(full_scale) else master_df
    best = pool.sort_values("MAE_vs_TA_avg").iloc[0]
    rid = best["run_id"]
    df = cached[rid].reset_index()
    ta_avg = ta_combined(df)
    ai = pd.to_numeric(df["AI Total Score"], errors="coerce")
    diff = ai - ta_avg
    valid = diff.notna()
    over = df[valid].assign(diff=diff[valid]).sort_values("diff", ascending=False).head(3)
    under = df[valid].assign(diff=diff[valid]).sort_values("diff").head(3)
    return rid, over, under


# -----------------------------------------------------------------------------
# 6. Plots
# -----------------------------------------------------------------------------

def make_plots(master_df: pd.DataFrame, cached: dict, per_cell_panels: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                         "font.size": 9, "axes.titlesize": 10})

    # --- Scatter: AI vs TA per model (top-9 by MAE so the figure is readable) ---
    top = master_df.sort_values("MAE_vs_TA_avg").head(9)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        d = cached[row["run_id"]].reset_index()
        ta = ta_combined(d)
        ai = pd.to_numeric(d["AI Total Score"], errors="coerce")
        v = ta.notna() & ai.notna()
        ax.scatter(ta[v], ai[v], s=6, alpha=0.4, color="#1f77b4", edgecolor="none")
        ax.plot([0, 35], [0, 35], "k--", lw=0.8, alpha=0.6)
        ax.set_title(f"{row['run_id']} · {row['model'][:18]}\nMAE={row['MAE_vs_TA_avg']:.2f} bias={row['bias_vs_TA_avg']:+.2f}", fontsize=8)
        ax.set_xlim(0, 36); ax.set_ylim(0, 36)
    for ax in axes[-1, :]: ax.set_xlabel("TA avg total")
    for ax in axes[:, 0]: ax.set_ylabel("AI total")
    fig.suptitle("AI vs TA total score (top-9 ablations by MAE)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "scatter_ai_vs_ta.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Per-question MAE bar chart (vs the two-TA per-question average) ---
    md = master_df.sort_values("MAE_vs_TA_avg")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(md))
    w = 0.26
    # Q1-Q3 only: Q4 is bonus-only, has no TA Score column, and is not part of
    # the 35-pt base scale, so it has no per-question MAE to plot.
    for i, q in enumerate([1, 2, 3]):
        ax.bar(x + (i - 1) * w, md[f"Q{q}_MAE_vs_TA"], w, label=f"Q{q}")
    ax.set_xticks(x)
    ax.set_xticklabels(md["run_id"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Per-question MAE vs two-TA average")
    ax.set_title("Per-question MAE vs two-TA average (per ablation, sorted by total MAE_vs_TA_avg)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_question_mae.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Cost vs quality Pareto (illustrative cost values) ---
    # Cost proxy: use a rough relative cost per 1k calls.
    # These are *approximate* and labeled as such in the markdown.
    cost_per_1k = {
        # Gemini: rough public API price proxies in USD per 1k calls
        # (placeholder relative scale; documented in summary).
        "gemini-flash-lite-latest": 0.5,
        "flash-lite": 0.5,
        "gemini-3-flash-preview": 5.0,
        "3-flash-preview": 5.0,
        "gemini-3.1-pro-preview": 15.0,
        "3.1-pro-preview": 15.0,
        "gemini-2.5-pro": 10.0,
        "2.5-pro": 10.0,
        # Local Qwen: GPU-hours/1k calls (rough proxy; small=cheap, big=$$$)
        "qwen2.5-coder-7b": 0.3,
        "qwen2.5-coder-14b": 0.7,
        "qwen2.5-coder-32b": 1.5,
        "qwen3-coder-30b-a3b": 1.2,
        "qwen3-coder-next": 4.0,
        # Big-model sweep (8x A100, FP8 for 480B). GPU-hours/1k calls.
        "qwen2.5-72b": 3.0,
        "qwen3-235b-a22b": 3.5,
        "qwen3-coder-480b": 5.5,
        # Non-Qwen open weights. Same rough proxy scale, NOT independently
        # measured: the eight Qwen anchors above are near-linear in total
        # parameters for dense models (cost ~ 0.042 * params_B reproduces
        # 7B->0.3, 14B->0.7, 32B->1.5, 72B->3.0), so dense entries are placed on
        # that line and MoE entries interpolated between the 30B-A3B (1.2) and
        # 235B-A22B (3.5) anchors on total size. Without these nine keys the
        # .map() below returned NaN and the dropna silently deleted 81 of the
        # 161 runs from the Pareto figure — every non-Qwen open-weight run.
        "llama-3.1-8b": 0.35,
        "llama-3.3-70b": 2.9,
        "glm-4-9b": 0.4,
        "glm-4-32b": 1.35,
        "glm-4.5-air": 2.0,          # 106B MoE, 12B active
        "gemma-3-12b": 0.5,
        "gemma-3-27b": 1.15,
        "mistral-small-24b": 1.0,
        "deepseek-coder-v2-lite": 0.65,  # 16B MoE, 2.4B active
    }
    md = master_df.copy()
    md["cost"] = md["model"].map(lambda m: cost_per_1k.get(m, np.nan))
    missing = sorted(md.loc[md["cost"].isna(), "model"].unique())
    if missing:
        # Loud, not silent: an unpriced model disappears from the figure, and a
        # figure that is missing runs looks exactly like a figure that is complete.
        n_dropped = int(md["cost"].isna().sum())
        warnings.warn(
            f"pareto_cost_vs_quality: {n_dropped} of {len(md)} runs dropped — no "
            f"cost_per_1k entry for {missing}. Add them or the figure understates "
            f"coverage.", RuntimeWarning, stacklevel=2,
        )
        print(f"   !! Pareto figure: dropped {n_dropped} runs, unpriced models: {missing}")
    md = md.dropna(subset=["cost"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, color, marker in [("gemini", "#d62728", "o"), ("local", "#2ca02c", "s")]:
        sub = md[md["source"] == src]
        ax.scatter(sub["cost"], sub["MAE_vs_TA_avg"], c=color, marker=marker, s=70,
                   label=src.capitalize(), alpha=0.85, edgecolor="k", linewidth=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r["run_id"], (r["cost"], r["MAE_vs_TA_avg"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Cost proxy (USD/1k calls — Gemini, or GPU-hours/1k — local)")
    ax.set_ylabel("MAE vs TA-avg total")
    ax.set_title("Cost–quality Pareto (cost values are rough order-of-magnitude proxies)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto_cost_vs_quality.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Variance histogram ---
    if per_cell_panels:
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 6, 31)
        colors = {"gemini-E01-t0.0": "#1f77b4", "gemini-E02-t0.5": "#ff7f0e",
                  "gemini-E03-t0.7": "#d62728",
                  "local-E01-qwen32b-t0.5": "#2ca02c",
                  "local-E02-qwen3coder-next-t0.5": "#9467bd"}
        for label, pdf in per_cell_panels.items():
            stds = pdf["std"].dropna()
            ax.hist(stds, bins=bins, alpha=0.45, label=label,
                    color=colors.get(label), histtype="stepfilled", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Per-cell stdev across N runs (same student × question)")
        ax.set_ylabel("# (student, question) cells")
        ax.set_title("Stability of repeated grading — distribution of per-cell stdev")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "variance_histogram.pdf", bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Markdown writers
# -----------------------------------------------------------------------------

def write_master_md(df: pd.DataFrame, floor: dict):
    md = df.sort_values("MAE_vs_TA_avg").copy()
    lines = [f"# Master comparison — all {len(md)} ablations\n",
             f"Human-grader floor (inter-grader MAE on {floor['n_pairs']} dual-graded "
             f"students, pooled across {floor['n_tas']} TAs / 10 fixed pairs): "
             f"**{floor['MAE']:.2f}** points. Any AI MAE below this is statistically "
             "indistinguishable from a second human TA.\n",
             "Sorted by MAE vs TA-average total score (ascending = better).\n",
             ""]
    cols = ["run_id", "source", "model", "strictness", "temperature",
            "n_graded", "n_failed",
            "mean_total_score", "std_total_score",
            "MAE_vs_TA_avg", "MAE_95CI", "bias_vs_TA_avg"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines += [header, sep]
    for _, r in md.iterrows():
        ci = f"[{r['MAE_CI95_lo']:.2f}, {r['MAE_CI95_hi']:.2f}]" if pd.notna(r.get("MAE_CI95_lo")) else "—"
        row = [str(r["run_id"]), r["source"], str(r["model"])[:28],
               str(r["strictness"]), f"{r['temperature']}",
               str(int(r["n_graded"])), str(int(r["n_failed"])),
               f"{r['mean_total_score']:.2f}", f"{r['std_total_score']:.2f}",
               f"{r['MAE_vs_TA_avg']:.2f}", ci, f"{r['bias_vs_TA_avg']:+.2f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Per-question MAE (vs the two-TA per-question average)\n")
    cols2 = ["run_id", "Q1_MAE_vs_TA", "Q2_MAE_vs_TA", "Q3_MAE_vs_TA",
             "Q1_mean", "Q2_mean", "Q3_mean", "Q4_mean"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "|".join(["---"] * len(cols2)) + "|")
    for _, r in md.iterrows():
        row = [str(r["run_id"])] + [f"{r[c]:.2f}" for c in cols2[1:]]
        lines.append("| " + " | ".join(row) + " |")
    (ANALYSIS / "computer_vision_master_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_human_floor_md(f: dict):
    lines = [
        "# Human-grader floor (inter-grader disagreement)\n",
        f"**Setup.** {f['n_tas']} TAs in total, organized into 10 fixed grading pairs "
        "(slot-1 TA always paired with the same slot-2 TA). Each student is graded "
        "exactly once by exactly one pair; each pair grades ~55–60 students. The "
        "`TA_1_ID` / `TA_2_ID` columns are therefore *slots*, not fixed identities — "
        "a signed `TA1 − TA2` bias would be uninterpretable, so we report only the "
        "magnitude of disagreement and correlation.\n",
        "## Pooled across all pairs\n",
        f"- **Dual-graded students:** {f['n_pairs']}",
        f"- **Inter-grader MAE on total score:** **{f['MAE']:.2f}** / 35  "
        f"(95% bootstrap CI [{f['MAE_CI95_lo']:.2f}, {f['MAE_CI95_hi']:.2f}])",
        f"- **Pearson r:** {f['pearson']:.3f}",
        f"- **Max single-paper disagreement:** {f['max_disagreement']:.1f} pts",
        f"- **Bonus inter-grader MAE:** {f['bonus_MAE']:.2f} (n={f['bonus_n']})",
        "",
        "### Distribution of |grader₁ − grader₂|",
        "",
        "| range (points) | # students |",
        "|---|---|",
    ]
    for interval, count in f["bins"].items():
        lines.append(f"| {interval} | {int(count)} |")

    lines += ["", "## Per-pair breakdown (10 pairs)\n",
              "Each row = one fixed pair of TAs (one slot-1, one slot-2). "
              "`mean_diff` is signed (slot-1 minus slot-2) and is **only** "
              "comparable *within* a pair — across pairs the slot labels mix different people.\n",
              "| pair | n | MAE | Pearson r | mean_diff (slot1 − slot2) |",
              "|---|---|---|---|---|"]
    for _, r in f["per_pair"].iterrows():
        lines.append(f"| {r['pair']} | {r['n']} | {r['MAE']:.2f} | "
                     f"{r['pearson_r']:.3f} | {r['mean_diff']:+.2f} |")

    lines += ["", "## Per-TA harshness (mean score awarded)\n",
              "How generous each individual TA is on the 0–35 scale, pooled across "
              "all submissions they graded (across both slots).\n",
              "| TA | n graded | mean score | std |",
              "|---|---|---|---|"]
    for ta, r in f["per_ta"].iterrows():
        lines.append(f"| {ta} | {int(r['count'])} | {r['mean']:.2f} | {r['std']:.2f} |")

    span = f["per_ta"]["mean"].max() - f["per_ta"]["mean"].min()
    harshest = f["per_ta"]["mean"].idxmin()
    softest = f["per_ta"]["mean"].idxmax()
    lines += [
        "",
        "## Interpretation\n",
        f"Two independent TAs grading the same 35-point exam disagree by **MAE "
        f"{f['MAE']:.2f}** on average; the maximum single-paper disagreement is "
        f"{f['max_disagreement']:.0f}/35. This is the *floor* — an AI grader with "
        "MAE at or below this value is, in aggregate, as accurate as a second human "
        "TA. It does not mean the AI agrees with either TA on any specific paper; "
        "only that its global error is no larger than the human noise channel.",
        "",
        f"Per-TA mean scores span **{span:.2f} pts** (harshest: {harshest} at "
        f"{f['per_ta']['mean'].min():.2f}; most generous: {softest} at "
        f"{f['per_ta']['mean'].max():.2f}). That spread is comparable to the "
        "inter-grader MAE itself and indicates that *which TA pair a student happens "
        "to be assigned to* is a non-trivial confound in the ground truth — the "
        "TA-average target the AI is judged against is itself noisy in a "
        "structured (pair-dependent) way.",
    ]
    (ANALYSIS / "computer_vision_human_floor.md").write_text("\n".join(lines), encoding="utf-8")


def write_variance_md(v_df: pd.DataFrame):
    lines = ["# Variance analysis (E rows: repeated grading)\n",
             "Each variance config grades the same student × question multiple times "
             "and we measure the sample stdev of the score across those repeats. "
             "Lower = more stable / reproducible.\n",
             ""]
    cols = list(v_df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in v_df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "**Key reading.** Gemini Flash-Lite at temp=0 is near-deterministic; at temp=0.7 "
        "the median per-cell stdev rises and the right tail (cells with std>1 point) "
        "fattens significantly. The Qwen3-Coder-Next configuration shows similar stability "
        "to mid-temperature Gemini, while Qwen2.5-Coder-32B at temp=0.5 is the most volatile "
        "config tested — repeated grading of the *same* notebook can disagree by 1+ points "
        "in a non-trivial fraction of cells.",
    ]
    (ANALYSIS / "computer_vision_variance.md").write_text("\n".join(lines), encoding="utf-8")


def write_failure_cases_md(rid: str, over: pd.DataFrame, under: pd.DataFrame, cached: dict):
    df = cached[rid]
    lines = [f"# Failure-mode vignettes\n",
             f"Best-config ablation: **{rid}**. We pick three students the AI scored "
             "**≥5 points above** the TA average and three it scored **≥5 below**.\n",
             "(Reasoning blobs truncated to ~600 chars; full text is in the result xlsx.)",
             ""]
    def truncate(s, n=600):
        if not isinstance(s, str): return ""
        s = " ".join(s.split())
        return s if len(s) <= n else s[:n] + "…"
    for label, sub in [("Over-graded by AI (AI > TA)", over), ("Under-graded by AI (AI < TA)", under)]:
        lines += [f"## {label}\n"]
        for _, r in sub.iterrows():
            n = int(r["Number"])
            ai_total = r.get("AI Total Score")
            ta_avg_v = ta_combined(pd.DataFrame([r])).iloc[0]
            diff = float(r["diff"])
            lines.append(f"### Student #{n}  (AI={ai_total:.1f}, TA-avg={ta_avg_v:.2f}, Δ={diff:+.2f})")
            sub_dir = SUBMISSIONS / str(n)
            lines.append(f"- Submission dir: `computer_vision_dataset/submissions_extracted/{n}/` ({'exists' if sub_dir.is_dir() else 'MISSING'})")
            full = df.loc[n]
            for q in [1, 2, 3, 4]:
                sc = full.get(f"AI Q{q} Score")
                rsg = full.get(f"AI Q{q} Reasoning")
                lines.append(f"  - **Q{q}** AI={sc}: {truncate(rsg)}")
            lines.append("")
    (ANALYSIS / "computer_vision_failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Paper-style markdown reports (regenerated from the same data)
# -----------------------------------------------------------------------------

def _pair_data(master_grades_df: pd.DataFrame, d01_df: pd.DataFrame) -> list[dict]:
    """Per-pair human MAE, D01 AI MAE, D01 AI bias. Sorted ascending by human MAE."""
    t1 = pd.to_numeric(master_grades_df["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(master_grades_df["TA 2 - Total Score (out of 35)"], errors="coerce")
    valid = t1.notna() & t2.notna()  # a 0/0 grade is a real dual grade
    sub = master_grades_df[valid].copy()
    sub["TA1_total"] = t1[valid]
    sub["TA2_total"] = t2[valid]
    sub["pair"] = sub.apply(
        lambda r: tuple(sorted([r["TA_1_ID"], r["TA_2_ID"]])), axis=1
    )
    sub = sub.set_index("Number")
    d01_indexed = d01_df if d01_df.index.name == "Number" else d01_df.set_index("Number")
    ai_total = pd.to_numeric(d01_indexed["AI Total Score"], errors="coerce")
    sub["AI_total"] = ai_total.reindex(sub.index)
    out = []
    for pair, g in sub.groupby("pair"):
        human_mae = float((g["TA1_total"] - g["TA2_total"]).abs().mean())
        ta_avg = (g["TA1_total"] + g["TA2_total"]) / 2.0
        ai_valid = g["AI_total"].notna()
        if ai_valid.sum() == 0:
            continue
        ai_mae = float((g.loc[ai_valid, "AI_total"] - ta_avg[ai_valid]).abs().mean())
        ai_bias = float((g.loc[ai_valid, "AI_total"] - ta_avg[ai_valid]).mean())
        out.append({
            "pair_label": f"{pair[1]} / {pair[0]}",
            "n": int(len(g)),
            "human_MAE": human_mae,
            "ai_MAE": ai_mae,
            "ai_bias": ai_bias,
        })
    out.sort(key=lambda d: d["human_MAE"])
    return out


def _per_ta_harshness(master_grades_df: pd.DataFrame) -> dict:
    """One-way ANOVA + Kruskal-Wallis across the 20 TAs (long form: 2 rows / dual-graded student)."""
    df = master_grades_df
    t1 = pd.to_numeric(df["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(df["TA 2 - Total Score (out of 35)"], errors="coerce")
    long = pd.concat([
        pd.DataFrame({"TA": df["TA_1_ID"], "score": t1}),
        pd.DataFrame({"TA": df["TA_2_ID"], "score": t2}),
    ]).dropna()
    by_ta = long.groupby("TA")["score"]
    means = by_ta.mean()
    groups = [g.values for _, g in by_ta]
    n_total = int(long.shape[0])
    n_groups = len(groups)
    f_stat, f_p = _scipy_stats.f_oneway(*groups)
    h_stat, h_p = _scipy_stats.kruskal(*groups)
    # eta^2 from one-way ANOVA: SS_between / SS_total
    grand = long["score"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((long["score"] - grand) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total else float("nan")
    return {
        "n_total": n_total,
        "n_groups": n_groups,
        "F": float(f_stat), "F_p": float(f_p),
        "H": float(h_stat), "H_p": float(h_p),
        "eta2": eta2,
        "df_between": n_groups - 1,
        "df_within": n_total - n_groups,
        "harshest_TA": str(means.idxmin()),
        "harshest_mean": float(means.min()),
        "softest_TA": str(means.idxmax()),
        "softest_mean": float(means.max()),
        "span": float(means.max() - means.min()),
        "per_ta_mean": means.sort_values(),
    }


def write_pair_level_ai_mae_md(master_grades_df: pd.DataFrame, cached: dict, master_df: pd.DataFrame):
    """Regenerated from data: per-pair MAE table, Pearson/Spearman, F-test, KW, eta^2."""
    if "D01" not in cached:
        return
    pair_rows = _pair_data(master_grades_df, cached["D01"].reset_index())
    xs = np.array([r["human_MAE"] for r in pair_rows])
    ys = np.array([r["ai_MAE"] for r in pair_rows])
    pearson_r, pearson_p = _scipy_stats.pearsonr(xs, ys)
    spearman_rho, spearman_p = _scipy_stats.spearmanr(xs, ys)
    harsh = _per_ta_harshness(master_grades_df)
    d01_row = master_df[master_df["run_id"] == "D01"].iloc[0]

    lines = [
        "# Pair-level AI error and per-TA harshness tests\n",
        "Two questions about the ground truth, both addressable from existing data:\n",
        "1. Does AI error track per-pair human disagreement? (i.e. does the AI *inherit* the per-pair noise structure, or *average it out*?)",
        "2. Is the per-TA harshness spread statistically real, or could it be chance?\n",
        f"Analyses use **D01** (gemini · 3-flash-preview, n={int(d01_row['n_graded'])}, MAE {d01_row['MAE_vs_TA_avg']:.2f}) as the AI side — the strongest full-scale configuration.\n",
        "## 1. Per-pair AI MAE vs per-pair human MAE\n",
        "| pair | n | human MAE | D01 AI MAE | D01 AI bias |",
        "|---|---|---|---|---|",
    ]
    for r in pair_rows:
        lines.append(
            f"| {r['pair_label']} | {r['n']} | {r['human_MAE']:.2f} | "
            f"{r['ai_MAE']:.2f} | {r['ai_bias']:+.2f} |"
        )
    noisiest = max(pair_rows, key=lambda r: r["human_MAE"])
    quietest = min(pair_rows, key=lambda r: r["human_MAE"])
    lines += [
        "",
        f"**Pearson r (human MAE × AI MAE) = {pearson_r:.3f}, p = {pearson_p:.3f}.**",
        f"**Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.3f}.**",
        "",
        f"The AI's per-pair MAE is essentially **{'uncorrelated' if pearson_p > 0.05 else 'correlated'}** with the per-pair human disagreement. "
        f"The noisiest human pair ({noisiest['pair_label']}, human MAE {noisiest['human_MAE']:.2f}) is graded at AI MAE {noisiest['ai_MAE']:.2f}; "
        f"the quietest human pair ({quietest['pair_label']}, {quietest['human_MAE']:.2f}) has AI MAE {quietest['ai_MAE']:.2f}.",
        "",
        "**Reading.** The AI is not inheriting the structure of pair-level human noise — "
        "it is *averaging across it*. The TA-average target is noisy, but the AI's residual error "
        "to that target is not modulated by which pair did the grading. This is a positive property: "
        "an AI grader trained without per-pair conditioning still produces a stable error profile.",
        "",
        "It does mean per-pair human MAE is the wrong upper bound for AI performance on any individual pair. "
        f"On the noisiest pair, the AI ({noisiest['ai_MAE']:.2f} MAE) is "
        f"{'*more consistent than the human pair itself*' if noisiest['ai_MAE'] < noisiest['human_MAE'] else 'still less consistent than the human pair'} "
        f"(human {noisiest['human_MAE']:.2f}).\n",
        "## 2. F-test on per-TA harshness\n",
        f"{harsh['n_groups']} TAs, each TA contributes the total scores they awarded across all "
        "submissions they graded (independent of slot). n = "
        f"{harsh['n_total']} (each dual-graded student contributes 2 (TA, score) rows).\n",
        f"- **One-way ANOVA:** F({harsh['df_between']}, {harsh['df_within']}) = **{harsh['F']:.3f}**, p = **{harsh['F_p']:.2g}**",
        f"- **Kruskal-Wallis (non-parametric):** H = {harsh['H']:.2f}, p = {harsh['H_p']:.2g}",
        f"- **η² = {harsh['eta2']:.3f}** — TA identity explains ≈{harsh['eta2']*100:.0f}% of the variance in awarded total scores.",
        "",
        f"Both tests {'reject' if harsh['F_p'] < 0.05 else 'fail to reject'} the null that all "
        f"{harsh['n_groups']} TAs have the same mean harshness. The {harsh['span']:.2f}-pt spread "
        f"(harshest TA {harsh['harshest_TA']} at {harsh['harshest_mean']:.2f}; most generous "
        f"{harsh['softest_TA']} at {harsh['softest_mean']:.2f}) is not a sampling artifact. "
        f"Effect size η² ≈ {harsh['eta2']:.2f}.",
        "",
        f"**Reading for the paper.** Per-TA harshness is a real, statistically distinguishable source of "
        "structured noise in the ground truth, *but* the AI does not appear to align with any one pair's "
        "behaviour — it sits roughly equidistant from all of them. The implication: the headline "
        "MAE-vs-TA-average metric understates how well the AI matches a *consensus* and overstates how "
        "well it would match any specific TA. Anyone consuming the AI's score would still need to know "
        "which pair the AI is being compared against to interpret a residual; the AI itself is not a "
        "per-pair surrogate.\n",
    ]
    (ANALYSIS / "computer_vision_pair_level_ai_mae.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# n100 restricted comparison
# -----------------------------------------------------------------------------

def _n100_metrics(rid: str, cached: dict, master_df: pd.DataFrame) -> dict | None:
    if rid not in cached:
        return None
    df = cached[rid].reset_index()
    df = df[df["Number"] <= 100]
    if df.empty:
        return None
    ta_avg = ta_combined(df).values
    ta_avg = pd.Series(ta_avg, index=df.index)
    ai_total = pd.to_numeric(df["AI Total Score"], errors="coerce")
    valid = ai_total.notna() & ta_avg.notna()
    diff = (ai_total[valid] - ta_avg[valid]).values
    if len(diff) == 0:
        return None
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    lo, hi = bootstrap_mae_ci(diff)
    return {"run_id": rid, "n": int(valid.sum()),
            "MAE": mae, "bias": bias, "ci_lo": lo, "ci_hi": hi}


def write_n100_restricted_md(master_df: pd.DataFrame, cached: dict):
    rows_spec = [
        ("D01",   "(baseline: sol+gd+bd, neutral)"),
        ("G04",   "rubric breakdown removed"),
        ("G03",   "+thinking"),
        ("G02",   "reference guidelines removed"),
        ("G01",   "reference solution removed"),
        ("M02",   "lenient persona"),
        ("M01",   "strict persona"),
    ]
    metrics = []
    for rid, label in rows_spec:
        m = _n100_metrics(rid, cached, master_df)
        if m is None:
            continue
        m["label"] = label
        metrics.append(m)
    # Sort: keep D01 first (baseline), then ascending MAE
    baseline = [m for m in metrics if m["run_id"] == "D01"]
    others = [m for m in metrics if m["run_id"] != "D01"]
    others.sort(key=lambda m: m["MAE"])
    metrics = baseline + others

    # D01 at full scale for the reference line
    d01_full_row = master_df[master_df["run_id"] == "D01"].iloc[0]

    lines = [
        "# D01 vs G/M series — same-subset comparison (Number ≤ 100)\n",
        "All G-series and M-series Gemini ablations ran on students 1-100 only "
        "(3-flash-preview is slow). Comparing them to D01 at full n=570 over-credits "
        "the G-series — sample variance alone drops MAE on smaller subsets. This file "
        "restricts D01 to the same Number ≤ 100 subset for an apples-to-apples comparison.\n",
        "Bootstrap 95% CI on n=100 (2000 resamples, seed 0).\n",
        "| run | prompt change vs D01 baseline | n | MAE | 95% CI | bias |",
        "|---|---|---|---|---|---|",
    ]
    for m in metrics:
        bold = "**" if m["run_id"] in ("D01", metrics[1]["run_id"] if len(metrics) > 1 else "") else ""
        run_disp = f"**{m['run_id']} (restricted)**" if m["run_id"] == "D01" else m["run_id"]
        lines.append(
            f"| {run_disp} | {m['label']} | {m['n']} | {bold}{m['MAE']:.2f}{bold} | "
            f"[{m['ci_lo']:.2f}, {m['ci_hi']:.2f}] | {m['bias']:+.2f} |"
        )
    lines += [
        "",
        f"For reference, D01 at full n={int(d01_full_row['n_graded'])}: "
        f"MAE **{d01_full_row['MAE_vs_TA_avg']:.2f}** "
        f"[{d01_full_row['MAE_CI95_lo']:.2f}, {d01_full_row['MAE_CI95_hi']:.2f}], "
        f"bias {d01_full_row['bias_vs_TA_avg']:+.2f}.\n",
        "## Reading\n",
        "- The full-table impression that G-series ablations beat D01 is a "
        "**sample-size artifact**. On the matched n=100 subset, D01 ties the best G "
        "variant within Monte-Carlo noise; removing the reference solution or guidelines hurts.",
        f"- D01's full-n MAE ({d01_full_row['MAE_vs_TA_avg']:.2f}) is "
        f"{'worse than' if d01_full_row['MAE_vs_TA_avg'] > metrics[0]['MAE'] else 'comparable to'} "
        f"its n=100 MAE ({metrics[0]['MAE']:.2f}) — "
        f"the first 100 students appear to be a slightly different subset (cleaner submissions "
        "and/or less noisy TA pairs), not an arena where prompt ablations win.",
        "- Persona effects (M01 strict, M02 lenient) reproduce on 3-flash-preview at the "
        "same magnitude as on Flash-Lite (C01 / C02): ±3–5 pt bias shift, no collapse.",
        "",
        "**Implication for the paper.** The \"common-sense recipe holds for closed models\" claim "
        "is *stronger* than the master table implied: on Gemini's best model, no single "
        "prompt-component removal helps, and the explicit recipe (solution + guidelines + breakdown, "
        "neutral persona) is at the joint optimum. Open-weights brittleness then becomes the *only* "
        "axis along which the recipe story varies — making the brittleness framing tighter.\n",
    ]
    (ANALYSIS / "computer_vision_n100_restricted_comparison.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# new_summary.md (paper-style summary, all numbers pulled from data)
# -----------------------------------------------------------------------------

def _mechanism_counts(rid: str, cached: dict) -> dict | None:
    """Selective-zeroing diagnostic for a strict-collapse ablation.

    Reports: of all students with AI Q1 Score == 0, what fraction also has
    AI Q2 Score > 0 (i.e. the model zeroed Q1 specifically rather than refusing
    to grade anything). High frac_selective = strict instruction selectively
    collapsed one question's structured score while the model continued grading
    the others normally.
    """
    if rid not in cached:
        return None
    df = cached[rid]
    if "AI Q1 Score" not in df.columns or "AI Q2 Score" not in df.columns:
        return None
    q1 = pd.to_numeric(df["AI Q1 Score"], errors="coerce")
    q2 = pd.to_numeric(df["AI Q2 Score"], errors="coerce")
    n_total = int(q1.notna().sum())
    q1_zero = (q1 == 0)
    n_zero = int(q1_zero.sum())
    if n_zero == 0:
        return {"n_total": n_total, "n_zero": 0, "n_selective": 0,
                "frac_zero": 0.0, "frac_selective": 0.0}
    n_selective = int((q1_zero & (q2 > 0)).sum())
    return {
        "n_total": n_total,
        "n_zero": n_zero,
        "n_selective": n_selective,
        "frac_zero": n_zero / n_total if n_total else 0.0,
        "frac_selective": n_selective / n_zero if n_zero else 0.0,
    }


def _row(master_df: pd.DataFrame, rid: str) -> pd.Series | None:
    r = master_df[master_df["run_id"] == rid]
    return r.iloc[0] if len(r) else None


def write_new_summary_md(master_df: pd.DataFrame, floor: dict,
                         v_df: pd.DataFrame, cached: dict):
    n_below = int((master_df["MAE_vs_TA_avg"] <= floor["MAE"]).sum())
    n_below_ci = int(
        (pd.to_numeric(master_df["MAE_CI95_hi"], errors="coerce") <= floor["MAE"]).sum()
    )
    full_scale = master_df[master_df["n_graded"] >= FULL_COHORT_MIN_N]\
        .sort_values("MAE_vs_TA_avg")
    best = full_scale.iloc[0]
    best_loc = full_scale[full_scale["source"] == "local"].iloc[0]
    best_gem = full_scale[full_scale["source"] == "gemini"].iloc[0]
    worst = full_scale.iloc[-1]

    def m(rid: str) -> str:
        r = _row(master_df, rid)
        return f"{r['MAE_vs_TA_avg']:.2f}" if r is not None else "n/a"

    def b(rid: str) -> str:
        r = _row(master_df, rid)
        return f"{r['bias_vs_TA_avg']:+.2f}" if r is not None else "n/a"

    def mean_total(rid: str) -> str:
        r = _row(master_df, rid)
        return f"{r['mean_total_score']:.2f}" if r is not None else "n/a"

    def ci(rid: str) -> str:
        r = _row(master_df, rid)
        if r is None or pd.isna(r.get("MAE_CI95_lo")):
            return "n/a"
        return f"[{r['MAE_CI95_lo']:.2f}, {r['MAE_CI95_hi']:.2f}]"

    def n(rid: str) -> str:
        r = _row(master_df, rid)
        return str(int(r["n_graded"])) if r is not None else "n/a"

    # Mechanism analysis: how each strict run actually fails. Computed for every
    # matched strict run, not for two hand-picked Qwen ones -- running it across
    # families is what showed there are three mechanisms here, not one.
    mech: dict[str, dict] = {}

    # Pair data
    master_grades = pd.read_excel(BASE / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx")
    pair_rows = _pair_data(master_grades, cached["D01"].reset_index()) if "D01" in cached else []
    xs = np.array([r["human_MAE"] for r in pair_rows]) if pair_rows else np.array([])
    ys = np.array([r["ai_MAE"] for r in pair_rows]) if pair_rows else np.array([])
    if len(xs) > 1:
        pearson_r, pearson_p = _scipy_stats.pearsonr(xs, ys)
        spearman_rho, spearman_p = _scipy_stats.spearmanr(xs, ys)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")
    harsh = _per_ta_harshness(master_grades)

    # E-series variance numbers from v_df
    v_idx = {row["config"]: row for _, row in v_df.iterrows()}

    L = []
    L.append(f"# Results: LLM-based auto-grading of a {int(best['n_valid_vs_TA'])}-student practical AI exam\n")
    # n_failed counts AI Total == 0 AND has-submission; on collapse and refusal runs
    # that is dominated by *intentional* zeroing by the model, not pipeline failure.
    # Report only the runs that actually graded, to keep the "no pipeline failures"
    # claim honest. This used to key off `MAE < 10`, which both over- and
    # under-counted: it swept in any run that merely scored badly, and it missed
    # collapses under 10. `behaviour` keys off the zero-rate directly.
    non_collapse = master_df[master_df["behaviour"] == "graded"]
    n_refusal = int((master_df["behaviour"] == "refusal").sum())
    n_near = int((master_df["behaviour"] == "near-refusal").sum())
    n_collapse = int((master_df["behaviour"] == "collapse").sum())
    # "All N produced complete grades" used to be an unconditional literal, and it
    # is no longer true. Compute it — but do not conflate the three reasons a run
    # can sit below 570. Some runs are DELIBERATELY small (the students-1-100
    # Gemini subsets, the n~50 temperature-variance repeats); lumping them in
    # with runs that lost students would both overstate the damage and hide it.
    n_students = int(master_df["n_valid_vs_TA"].max())
    full = master_df[master_df["n_valid_vs_TA"] >= FULL_COHORT_MIN_N]
    subsets = master_df[master_df["n_valid_vs_TA"] < FULL_COHORT_MIN_N]
    short = full[full["n_valid_vs_TA"] < n_students]
    completeness = (
        f"All {len(full)} full-cohort ablations graded all {n_students} students."
        if short.empty else
        f"{len(full) - len(short)} of {len(full)} full-cohort ablations graded all "
        f"{n_students} students; {len(short)} lost between "
        f"{n_students - int(short['n_valid_vs_TA'].max())} and "
        f"{n_students - int(short['n_valid_vs_TA'].min())} students to calls that "
        f"returned no usable score."
    )
    if len(subsets):
        completeness += (
            f" A further {len(subsets)} runs are deliberate small-n designs "
            f"(n={int(subsets['n_valid_vs_TA'].min())}–"
            f"{int(subsets['n_valid_vs_TA'].max())}) and are excluded from that count."
        )
    L.append(
        f"A {len(master_df)}-ablation study ({family_breakdown(master_df)}) of "
        f"LLM-based code grading on a 35-pt KAUST practical AI exam ({floor['n_pairs']} dual-graded students, "
        f"4 questions, {harsh['n_groups']} TAs in 10 fixed grading pairs). {completeness} "
        f"By behaviour: {n_collapse} runs **collapse** (grade harshly but still rank students), "
        f"{n_refusal} **refuse** outright (zero-rate ≥{REFUSAL_ZERO_RATE:.0%}, stdev "
        f"<{REFUSAL_STDEV}) and {n_near} sit in between. Excluding all three — where the model "
        f"zeroes scores by design rather than by failure — pipeline failures sum to "
        f"{int(non_collapse['n_failed'].sum())} across the remaining {len(non_collapse)} "
        f"configurations.\n"
    )
    L.append(
        "\n> **Refusal is not collapse.** Both saturate MAE near 26.04 — the distance from the "
        "TA mean — so MAE cannot tell them apart; `zero_rate`, `std_total_score` and "
        "`AI_TA_pearson_r` can. A collapsed grader still correlates with the TAs (r 0.47–0.78); "
        "a refusing one does not correlate at all (r undefined or ≈0). Reporting them under one "
        "heading conflates *grades harshly* with *does not grade*.\n"
    )

    L.append("## Headlines\n")
    L.append(
        f"- **Human-grader floor**: inter-grader MAE **{floor['MAE']:.2f} / 35** on {floor['n_pairs']} "
        f"dual-graded students (95% bootstrap CI [{floor['MAE_CI95_lo']:.2f}, {floor['MAE_CI95_hi']:.2f}]; "
        f"Pearson r = {floor['pearson']:.3f}; max single-paper disagreement {floor['max_disagreement']:.1f} pts)."
    )
    L.append(
        f"- **{n_below} / {len(master_df)} ablations match or undercut the floor by point estimate**; "
        f"**{n_below_ci} configurations have CI upper-bound under the floor** — they are statistically "
        "indistinguishable from a second human grader on aggregate error."
    )
    L.append(
        f"- **Best full-scale (n≥500) configuration**: **{best['run_id']}** "
        f"(`{best['model']}`, {best['strictness']}, t={best['temperature']}), "
        f"MAE **{best['MAE_vs_TA_avg']:.2f}** "
        f"[{best['MAE_CI95_lo']:.2f}, {best['MAE_CI95_hi']:.2f}], bias {best['bias_vs_TA_avg']:+.2f}."
    )
    L.append(
        f"- **Best open-weights full-scale configuration**: **{best_loc['run_id']}** "
        f"(`{best_loc['model']}` + {best_loc['strictness']}), "
        f"MAE **{best_loc['MAE_vs_TA_avg']:.2f}** "
        f"[{best_loc['MAE_CI95_lo']:.2f}, {best_loc['MAE_CI95_hi']:.2f}], bias {best_loc['bias_vs_TA_avg']:+.2f} — "
        f"within **{best_loc['MAE_vs_TA_avg'] - best_gem['MAE_vs_TA_avg']:.2f} MAE pts** of the best closed model."
    )
    L.append(
        f"- **Worst configuration**: **{worst['run_id']}** (`{worst['model']}` + {worst['strictness']}), "
        f"MAE **{worst['MAE_vs_TA_avg']:.2f}** "
        f"[{worst['MAE_CI95_lo']:.2f}, {worst['MAE_CI95_hi']:.2f}], "
        f"mean awarded total {worst['mean_total_score']:.2f} / 35."
    )
    # Brittleness statement, measured across every family present rather than
    # asserted from a hand-picked run list. This bullet used to name four Qwen
    # run-ids and then generalise the conclusion to open weights in prose; with
    # six open-weight families in the study that generalisation is now testable,
    # and one half of it does not survive (see the wording bullet below).
    pairs = {p: matched_persona_pairs(master_df, p)
             for p in ("strict", "rigorous", "exacting")}
    strict_pairs = pairs["strict"]

    def _open(df):
        return df[~df["family"].isin(CLOSED_FAMILIES)] if len(df) else df

    def _broke(df):
        return df[df["test_behaviour"] != "graded"] if len(df) else df

    if len(strict_pairs):
        op = _open(strict_pairs)
        broke = _broke(op)
        gave_up = op[op["test_behaviour"].isin(("refusal", "near-refusal"))]
        held = op[op["test_behaviour"] == "graded"].sort_values("test_MAE")
        gem = strict_pairs[strict_pairs["family"] == "Gemini"]
        nm = lambda r: f"{r.model} {r.test_rid} {r.test_MAE:.2f}"  # noqa: E731
        nms = lambda r: (f"{r.model} ({r.params_b:.0f}B) {r.test_MAE:.2f}"  # noqa: E731
                         if pd.notna(r.params_b) else f"{r.model} {r.test_MAE:.2f}")

        bullet = (
            f"- **The strict persona breaks every open-weight family tested — the collapse is "
            f"not a Qwen artefact.** Across **{len(op)} config-matched strict/neutral pairs "
            f"spanning {op['family'].nunique()} open-weight families** (identical prompt "
            f"components, zero-shot, t=0, n≥{FULL_COHORT_MIN_N}; only the persona sentence "
            f"differs), **{len(broke)} leave the graded band**, MAE rising "
            f"×{op['ratio'].min():.2f}–×{op['ratio'].max():.2f}. "
            f"{len(gave_up)} stop grading altogether "
            f"({'; '.join(nm(r) for r in gave_up.itertuples())}) — zero to ~every student. "
            f"Only {len(held)} hold ({', '.join(nms(r) for r in held.itertuples())})"
        )
        # Worth stating explicitly when it is true, because it is the one place the
        # Qwen-centred framing was right for the wrong reason: Qwen survives here
        # not as a family trait but because it is the only family with rungs at the
        # top of the ladder. Note the holders are NOT simply "the big ones" -- see
        # the size ladder in section 3, which unpacks the smallest of them.
        if len(held) > 1 and held["family"].nunique() == 1:
            bullet += (f" — and all {len(held)} are **{held['family'].iloc[0]}**, the only "
                       f"family with rungs above 106B. Family and scale are therefore "
                       f"confounded at the robust end and cannot be separated by this data")
        bullet += "."
        if len(gem):
            g0 = gem.iloc[0]
            bullet += (f" Gemini {g0['model']} shifts ×{g0['ratio']:.2f} "
                       f"({g0['test_MAE']:.2f}) and never leaves the band.")
        L.append(bullet + "\n")

        # Why "strict" breaks graders when "rigorous" and "exacting" do not.
        #
        # The tempting reading of that gap is lexical -- that the token "strict"
        # is itself the trigger. It is NOT, and saying so would contradict the
        # two series built to attribute it. The three presets are not minimal
        # pairs: only `strict` carries the sentences "Award the MINIMUM defensible
        # score..." and "Never give partial credit..." (STRICTNESS_PRESETS,
        # computer_vision_grade_with_local.py:104-118). The difference is policy, not vocabulary.
        alt = {p: _open(pairs[p]) for p in ("rigorous", "exacting")}
        if all(len(v) for v in alt.values()):
            L.append(
                f"- **The damage is carried by a policy sentence, not by the adjective.** The "
                f"same models under the other strict-flavoured presets barely move: "
                + "; ".join(
                    f"**{len(_broke(v))}/{len(v)}** leave the band on \"{p}\" "
                    f"(max ×{v['ratio'].max():.2f})" for p, v in alt.items())
                + f" — against **{len(broke)}/{len(op)}** on \"strict\" "
                  f"(max ×{op['ratio'].max():.2f}). That gap is **not** evidence that the word "
                  "\"strict\" is the trigger: the three presets are not minimal pairs. Only "
                  "`strict` instructs the model to \"award the MINIMUM defensible score\" and "
                  "\"never give partial credit\"; `rigorous` and `exacting` ask for rubric "
                  "precision without either clause. The two attribution series below separate "
                  "the headword from the policy and put the effect squarely on the policy.\n"
            )

        # Attribution: headword-varied (policy fixed) vs policy-varied (frame fixed).
        def _persona_mae(keys):
            """{model: {persona: MAE}} for models that ran every persona in `keys`."""
            sel = master_df[master_df["n_valid_vs_TA"] >= FULL_COHORT_MIN_N]
            for k, v in DEFAULT_PROMPT_CONFIG.items():
                if k in sel.columns:
                    sel = sel[sel[k] == v]
            out = {}
            for model, g in sel.groupby("model"):
                v = {r.strictness: float(r.MAE_vs_TA_avg) for r in g.itertuples()}
                if all(k in v for k in keys):
                    out[model] = v
            return out

        head = _persona_mae(("mpstrict", "mprigorous", "mpfair"))
        if head:
            spreads = {mo: max(v[k] for k in ("mpstrict", "mprigorous", "mpfair"))
                           - min(v[k] for k in ("mpstrict", "mprigorous", "mpfair"))
                       for mo, v in head.items()}
            worst_fair = max(head.items(), key=lambda kv: kv[1]["mpfair"])
            L.append(
                f"- **Swapping the headword changes almost nothing.** Holding the strict "
                f"preset's two policy sentences verbatim and varying only the adjective "
                f"(STRICT / RIGOROUS / FAIR) across {len(head)} models moves MAE by "
                f"{min(spreads.values()):.2f}–{max(spreads.values()):.2f} points. Calling the "
                f"model a **FAIR** teaching assistant while keeping the harsh policy leaves "
                f"{worst_fair[0]} at {worst_fair[1]['mpfair']:.2f} — still not grading.\n"
            )

        cells = ("mpframe", "mpnoclause", "mps2only", "mpharsh")
        pol = _persona_mae(cells)
        if pol:
            L.append(
                "- **Removing the policy sentences recovers the grader.** A 2×2 over the two "
                "sentences, with the \"You are a HARSH teaching assistant\" frame held fixed "
                "(S1 = award the minimum defensible score; S2 = never give partial credit):\n"
            )
            L.append("| model | frame only | + S1 | + S2 | both (= strict) |")
            L.append("|---|---:|---:|---:|---:|")
            for mo, v in sorted(pol.items(), key=lambda kv: -kv[1]["mpharsh"]):
                L.append(f"| {mo} | {v['mpframe']:.2f} | {v['mpnoclause']:.2f} | "
                         f"{v['mps2only']:.2f} | **{v['mpharsh']:.2f}** |")
            L.append("")
            drops = {mo: v["mpharsh"] - v["mpframe"] for mo, v in pol.items()}
            big = max(drops.items(), key=lambda kv: kv[1])
            frame_only = ", ".join(f"{mo} {v['mpframe']:.2f}" for mo, v in pol.items())
            L.append(
                f"  The harsh *frame* alone is nearly harmless ({frame_only}); "
                f"adding the sentences is what destroys the grader — {big[0]} moves "
                f"{pol[big[0]]['mpframe']:.2f} → {pol[big[0]]['mpharsh']:.2f}, a "
                f"{big[1]:.2f}-point swing from two sentences of policy. **\"Be strict\" is not "
                "what breaks the grader; \"never give partial credit\" is** — an instruction that "
                "directly contradicts the partial-credit scale the rubric scaffold mandates.\n"
            )

    L.append("## Per-axis findings\n")
    L.append("### 1. Prompt-component ablations (B-series, Gemini flash-lite)\n")
    L.append("| ablation | change vs A01 | MAE | bias |")
    L.append("|---|---|---|---|")
    b_rows = [("A01", "baseline"), ("B01", "reference solution removed"),
              ("B02", "guidelines removed"), ("B03", "thinking mode ON"),
              ("B04", "rubric breakdown removed")]
    best_b = min((rid for rid, _ in b_rows if _row(master_df, rid) is not None),
                 key=lambda r: float(m(r)))
    for rid, lbl in b_rows:
        if _row(master_df, rid) is None: continue
        mae_v = m(rid); bold = "**" if rid == best_b else ""
        L.append(f"| {rid} | {lbl} | {bold}{mae_v}{bold} | {b(rid)} |")
    L.append("")
    L.append(
        f"Removing the reference solution flips the model from under-grading to over-grading "
        f"(bias {b('A01')} → {b('B01')}). Removing guidelines or the rubric breakdown shaves "
        "~0.1 MAE — small. The only large move is **thinking mode** (B03), which brings flash-lite "
        f"to MAE {m('B03')} at ~1/10 the cost of the 3-flash-preview tier.\n"
    )

    L.append("### 2. Prompt ablations replayed on the best Gemini model (G-series, n=100)\n")
    L.append(
        "The original master comparison appeared to show G03/G04 beating D01. That is a "
        "**sample-size artifact**: G-series ran on Number ≤ 100 only because 3-flash-preview is slow. "
        "Restricting D01 to the same subset:\n"
    )
    L.append("| run | change vs D01 baseline | n | MAE (n≤100) | bias |")
    L.append("|---|---|---|---|---|")
    g_spec = [("D01", "(baseline: sol+gd+bd, neutral)"),
              ("G04", "rubric breakdown removed"),
              ("G03", "+thinking"),
              ("G02", "guidelines removed"),
              ("G01", "reference solution removed")]
    g_metrics = []
    for rid, lbl in g_spec:
        mm = _n100_metrics(rid, cached, master_df)
        if mm is None: continue
        mm["label"] = lbl
        g_metrics.append(mm)
    g_baseline = [r for r in g_metrics if r["run_id"] == "D01"]
    g_others = sorted([r for r in g_metrics if r["run_id"] != "D01"], key=lambda x: x["MAE"])
    g_metrics = g_baseline + g_others
    for mm in g_metrics:
        bold = "**" if (mm["run_id"] == "D01" or (g_others and mm["run_id"] == g_others[0]["run_id"])) else ""
        disp = f"**{mm['run_id']} (restricted)**" if mm["run_id"] == "D01" else mm["run_id"]
        L.append(f"| {disp} | {mm['label']} | {mm['n']} | {bold}{mm['MAE']:.2f}{bold} "
                 f"[{mm['ci_lo']:.2f}, {mm['ci_hi']:.2f}] | {mm['bias']:+.2f} |")
    L.append("")
    L.append(
        "**On the best closed model, no single prompt-component removal helps.** The baseline recipe "
        "(reference solution + guidelines + rubric breakdown, neutral persona) sits at the joint "
        "optimum on 3-flash-preview within MC noise. Source: "
        "[n100_restricted_comparison.md](computer_vision_n100_restricted_comparison.md).\n"
    )

    L.append("### 3. Strictness persona\n")
    L.append("Gemini personas behave as a clean ±3–5 pt calibration knob:\n")
    L.append("| run | model | strictness | MAE | bias | mean total |")
    L.append("|---|---|---|---|---|---|")
    for rid in ["A01", "C01", "C02", "M01", "M02"]:
        r = _row(master_df, rid)
        if r is None: continue
        ext = " (n=100)" if int(r["n_graded"]) <= 110 else ""
        L.append(
            f"| {rid} | {r['model']} | {r['strictness']}{ext} | "
            f"{r['MAE_vs_TA_avg']:.2f} | {r['bias_vs_TA_avg']:+.2f} | "
            f"{r['mean_total_score']:.2f} |"
        )
    L.append("")
    L.append(
        "On open weights the same persona is **catastrophic**, and it is catastrophic in every "
        "family. The ladder below is ordered by parameter count, not by family, so the size "
        "claim can be read off it directly instead of being asserted:\n"
    )
    if len(strict_pairs):
        ladder = strict_pairs.sort_values(
            ["params_b", "model"], na_position="last").reset_index(drop=True)
        L.append("| params | family | model | neutral | **strict** | ratio | strict mean | behaviour |")
        L.append("|---:|---|---|---:|---:|---:|---:|---|")
        for r in ladder.itertuples():
            size = f"{r.params_b:.0f}B" if pd.notna(r.params_b) else "?"
            mark = "" if r.test_behaviour == "graded" else "**"
            L.append(
                f"| {size} | {r.family} | {r.model} | {r.baseline_MAE:.2f} | "
                f"{mark}{r.test_MAE:.2f}{mark} | ×{r.ratio:.2f} | {r.test_mean:.2f} | "
                f"{r.test_behaviour} |"
            )
        L.append("")

        op = _open(strict_pairs)
        sized = op.dropna(subset=["params_b"])
        big = sized[sized["params_b"] > 100]
        unsized = op[op["params_b"].isna()]
        held = op[op["test_behaviour"] == "graded"].sort_values("params_b")
        worst2 = sized.nlargest(2, "ratio")
        # Spearman, not Pearson: the question is whether rank-order in size tracks
        # rank-order in damage, and params_b spans 7 -> 480 (two decades).
        rho = (_scipy_stats.spearmanr(sized["params_b"], sized["ratio"]).statistic
               if len(sized) > 2 else float("nan"))
        # Only prefix the size when the model name does not already carry it, so we
        # get "16B deepseek-coder-v2-lite" but plain "glm-4-9b", not "9B glm-4-9b".
        def sz(r):
            if pd.isna(r.params_b):
                return str(r.model)
            tag = f"{r.params_b:.0f}b"
            return str(r.model) if tag in str(r.model).lower() else f"{tag.upper()} {r.model}"
        L.append(
            f"**Size is a weak predictor below ~100B, and above it size and family cannot be "
            f"separated.** Rank correlation between parameter count and strict/neutral MAE ratio "
            f"across the {len(sized)} sized open-weight models is ρ = {rho:+.2f} — no usable "
            f"trend. The two hardest failures in the study are "
            f"{' and '.join(sz(r) for r in worst2.itertuples())}, both of which stop grading "
            f"entirely, while {sz(big.nsmallest(1, 'params_b').iloc[0])} still collapses to MAE "
            f"{big.nsmallest(1, 'params_b')['test_MAE'].iloc[0]:.2f}. "
            f"The earlier reading — that brittleness is a property of *small-to-mid* open-weights "
            f"instruction following — is **not supported once more than one family is on the "
            f"ladder**: it was the shape of the Qwen curve, not of open weights. "
            f"(The Gemini row is carried as a closed-model control, and is the only non-open "
            f"entry in the table.)"
            + (f" {', '.join(unsized['model'])} is omitted from the size claim only: its "
               f"parameter count is not verifiable from the local checkpoint, though it "
               f"collapses like the rest." if len(unsized) else "")
            + "\n"
        )
        # Which models "hold" is the claim most likely to be misread, because the
        # holders are NOT the top rungs -- one of them is the smallest model on the
        # ladder, sitting a rounding error under the cutoff. Say so explicitly.
        if len(held):
            top = held[held["params_b"] > 100]
            edge = held[held["test_MAE"] >= COLLAPSE_MAE - 0.5]
            txt = (f"**Which models hold, and why that is not simply \"the big ones\".** "
                   f"{len(held)} open-weight models stay in the graded band under strict: "
                   f"{', '.join(f'{sz(r)} ({r.test_MAE:.2f}, ×{r.ratio:.2f})' for r in held.itertuples())}. "
                   f"{len(top)} of those are the two largest models in the study and both are "
                   f"{top['family'].iloc[0] if len(top) and top['family'].nunique() == 1 else 'one family'}, "
                   f"so scale and family are confounded at the robust end and this data cannot "
                   f"separate them.")
            if len(edge):
                e = edge.iloc[0]
                txt += (f" The remaining holder is the *smallest* model on the ladder and it holds "
                        f"only just: {e['model']} at {e['test_MAE']:.2f} clears the {COLLAPSE_MAE:.2f} "
                        f"cutoff by {COLLAPSE_MAE - e['test_MAE']:.2f} MAE. It also has one of the "
                        f"worst neutral baselines here ({e['baseline_MAE']:.2f}), and a model that "
                        f"already grades poorly has less room to degrade — so its low ratio "
                        f"(×{e['ratio']:.2f}) is partly a floor effect, not evidence of robustness. "
                        f"Read the strict column, not the ratio, for this row.")
            L.append(txt + "\n")

    if len(strict_pairs):
        for r in strict_pairs.itertuples():
            mc = _mechanism_counts(r.test_rid, cached)
            if mc:
                mech[r.test_rid] = mc
    if mech:
        L.append("#### Mechanism: three different failures, one MAE range\n")
        L.append(
            "Of the students a run scored 0 on Q1, what fraction *also* got a non-zero Q2? "
            "The answer separates failures that look identical in the MAE column. This "
            "diagnostic was previously reported on two Qwen runs and read as a single "
            "mechanism; run across all "
            f"{len(mech)} matched strict runs it resolves into **three**:\n"
        )
        MECH_ORDER = ("blanket zeroing", "selective field collapse", "uniform severity")
        by_mech: dict[str, list] = {}
        for r in strict_pairs.itertuples():
            if r.test_rid in mech:
                by_mech.setdefault(mechanism_class(mech[r.test_rid]), []).append(r)
        # Group the table by mechanism explicitly. It came out grouped anyway when
        # sorted by ratio, but only because ratio happens to track mechanism here --
        # that is a coincidence of this data, not a property to rely on.
        L.append("| run | family | model | MAE | Q1 = 0 | of which Q2 > 0 | mechanism |")
        L.append("|---|---|---|---:|---:|---:|---|")
        for cls in MECH_ORDER:
            for r in sorted(by_mech.get(cls, []), key=lambda x: -x.test_MAE):
                mc = mech[r.test_rid]
                L.append(
                    f"| {r.test_rid} | {r.family} | {r.model} | {r.test_MAE:.2f} | "
                    f"{mc['n_zero']} / {mc['n_total']} ({mc['frac_zero']*100:.0f}%) | "
                    f"{mc['n_selective']} ({mc['frac_selective']*100:.0f}%) | {cls} |"
                )
        L.append("")

        blanket = by_mech.get("blanket zeroing", [])
        selective = by_mech.get("selective field collapse", [])
        uniform = by_mech.get("uniform severity", [])
        names = lambda xs: ", ".join(x.model for x in xs)  # noqa: E731

        def _rs(xs):
            v = [_row(master_df, x.test_rid)["AI_TA_pearson_r"] for x in xs]
            v = [x for x in v if pd.notna(x)]
            if not v:
                return "n/a"
            return f"{v[0]:.2f}" if min(v) == max(v) else f"{min(v):.2f}–{max(v):.2f}"

        # The mechanism is "when it zeroes, it zeroes everything" -- which is NOT the
        # same as "it zeroed everyone". Three of these are refusals by the behaviour
        # classifier; glm-4.5-air does it to a majority and still grades the rest, so
        # calling the whole bucket "the refusals" would misdescribe that run.
        # Split on run_id, not on `x not in refusals`: itertuples rows carry
        # params_b, which is NaN for some models, and NaN != NaN makes tuple
        # membership silently wrong for exactly those rows.
        refusals = [x for x in blanket if x.test_behaviour in ("refusal", "near-refusal")]
        _ref_ids = {x.test_rid for x in refusals}
        partial = [x for x in blanket if x.test_rid not in _ref_ids]
        L.append(
            f"- **Blanket zeroing** ({len(blanket)} runs: {names(blanket)}) — when these models "
            "zero a student they zero the whole submission; Q2 almost never survives a zeroed "
            f"Q1. In {len(refusals)} of them ({names(refusals)}) that happens to ~every student, "
            "and the behaviour classifier calls them refusals: the model is not grading harshly, "
            f"it is not grading. Their MAE near "
            f"{max((x.test_MAE for x in refusals), default=float('nan')):.1f} is just the "
            "distance from the TA mean — a ceiling artefact, not a severity measurement."
            + (f" {names(partial)} is the intermediate case: it blanket-zeroes "
               f"{mech[partial[0].test_rid]['frac_zero']*100:.0f}% of students and still grades "
               f"the rest (r = {_rs(partial)}), so it is a collapse, not a refusal."
               if partial else "") + "\n"
            f"- **Selective field collapse** ({len(selective)} runs: {names(selective)}) — Q1 is "
            "zeroed on a large minority of students while Q2 keeps receiving real marks. This is "
            "the instruction-following failure on a conflicting prompt (\"be strict\" vs. \"award "
            "partial credit per rubric\") that dismantles individual structured fields while the "
            f"rest of the response continues normally (r = {_rs(selective)}).\n"
            f"- **Uniform severity** ({len(uniform)} runs: {names(uniform)}) — essentially no "
            "zeroing at all; the model simply marks everything down. These still rank students "
            f"well (r = {_rs(uniform)}), so they are miscalibrated rather than broken — but note "
            f"that {sum(1 for x in uniform if x.test_behaviour != 'graded')} of them are still "
            "far enough off to leave the graded band, which is what makes MAE alone unable to "
            "tell this bucket from the other two.\n"
        )
        L.append(
            "Reporting these three under one heading would conflate \"awards nothing\" with "
            "\"zeroes one field\" with \"grades everything harshly\" — three different claims "
            "about what a strict instruction does to a model, and the earlier two-run version of "
            "this table asserted the middle one for all of them. For prose evidence of the "
            "selective case (rationale text awarding partial credit while the structured Q1 "
            "score is 0) see [failure_cases.md](computer_vision_failure_cases.md).\n"
        )

    L.append("### 4. Model size and family\n")
    L.append(
        f"**Gemini.** D01 (3-Flash-Preview) is the single best full-scale config ({m('D01')}). "
        f"D02 (3.1-Pro-Preview) is close ({m('D02')}). D03 (2.5-Pro, older generation) lags at "
        f"{m('D03')}. **Newer-generation Flash beats older-generation Pro.**\n"
    )
    a07, a14, a32 = m("L-A07"), m("L-A14"), m("L-A32")
    a07_b, a14_b, a32_b = b("L-A07"), b("L-A14"), b("L-A32")
    L.append(
        f"**Qwen2.5-Coder size sweep — non-monotonic.** 7B → 14B improves ({a07} → {a14}), but "
        f"**14B → 32B regresses badly** ({a14} → {a32}; bias {a14_b} → {a32_b}). "
        "The 32B baseline systematically under-grades.\n"
    )
    L.append(
        f"L-B01 (32B with the reference solution *removed*) drops 32B's MAE from {a32} → {m('L-B01')} — "
        "consistent with **over-anchoring on the reference solution**: with the solution in-context, "
        "the 32B model penalises any divergence; without it, it grades reasonably.\n"
    )
    L.append(
        f"**Qwen3 family.** Qwen3-30B-A3B (MoE, ~3B active) is the strongest open baseline "
        f"(L-A30M, MAE {m('L-A30M')}). Qwen3-Coder-Next (80B MoE) underperforms despite being larger "
        f"(L-ANxt {m('L-ANxt')}, bias {b('L-ANxt')}). **Thinking mode is a no-op on Qwen3 MoE**: "
        f"L-A30M vs L-D30M (same model ± thinking) {m('L-A30M')} vs {m('L-D30M')}.\n"
    )

    L.append("### 5. Few-shot prompting (K-series)\n")
    L.append("| run | model | MAE | bias |")
    L.append("|---|---|---|---|")
    for rid in ["A01", "K01", "L-A30M", "L-K01"]:
        r = _row(master_df, rid)
        if r is None: continue
        bold = "**" if rid == "L-K01" else ""
        L.append(f"| {rid} | {r['model']} | {bold}{r['MAE_vs_TA_avg']:.2f}{bold} | {r['bias_vs_TA_avg']:+.2f} |")
    L.append("")
    a01_mae = _row(master_df, "A01")
    k01_mae = _row(master_df, "K01")
    la30 = _row(master_df, "L-A30M")
    lk01 = _row(master_df, "L-K01")
    if all(x is not None for x in [a01_mae, k01_mae, la30, lk01]):
        d_closed = a01_mae["MAE_vs_TA_avg"] - k01_mae["MAE_vs_TA_avg"]
        d_open = la30["MAE_vs_TA_avg"] - lk01["MAE_vs_TA_avg"]
        L.append(
            f"**Few-shot helps the closed model by {d_closed:+.2f} MAE** (A01 {m('A01')} → K01 {m('K01')}) "
            f"**and the open-weights model by {d_open:+.2f} MAE** (L-A30M {m('L-A30M')} → L-K01 {m('L-K01')}). "
            "The two demonstrations were drawn from the highest-agreement TA pair (TA_6 / TA_16, "
            "inter-grader MAE ~0.85) with scores and rationales taken from the D01 configuration; "
            "this should be disclosed in the methods section.\n"
        )

    L.append("### 6. Temperature / variance (E-series)\n")
    L.append("| config | n_cells | median per-cell std | % cells with std > 1 pt |")
    L.append("|---|---|---|---|")
    for cfg in ["gemini-E01-t0.0", "gemini-E02-t0.5", "gemini-E03-t0.7",
                "local-E01-qwen32b-t0.5", "local-E02-qwen3coder-next-t0.5"]:
        if cfg in v_idx:
            row = v_idx[cfg]
            L.append(
                f"| {cfg} | {int(row['n_cells'])} | {row['median_cell_std']:.3f} | "
                f"{row['pct_cells_std_gt_1']:.1f}% |"
            )
    L.append("")
    L.append(
        "**Gemini at t=0 is fully deterministic** (100% of (student, question) cells have stdev = 0 "
        "across 5 reruns). At t=0.7 the right tail fattens substantially. Open-weights variance at "
        "t=0.5 is comparable to Gemini at t=0.7. **Production recommendation: t=0, single call.**\n"
    )

    # ---------------------------------------------------------- Cost vs quality
    L.append("## Cost–quality ([../media/pareto_cost_vs_quality.pdf](../media/pareto_cost_vs_quality.pdf))\n")
    lr01 = _row(master_df, "L-R01")
    lr04 = _row(master_df, "L-R04")
    lq01 = _row(master_df, "L-Q01")
    ln01 = _row(master_df, "L-N01")
    la30m = _row(master_df, "L-A30M")
    b03 = _row(master_df, "B03")
    d01 = _row(master_df, "D01")
    parts = []
    parts.append(
        "On the Gemini side the Pareto frontier is just two points: **Flash-Lite + thinking (B03)** "
        f"at MAE {m('B03')} for ~$0.5/1k calls, and **3-Flash-Preview (D01)** at MAE {m('D01')} for "
        "roughly an order of magnitude more. The Pro tier (D02 / D03) is dominated."
    )
    if lr01 is not None and lr04 is not None:
        parts.append(
            "On the open-weights side the Pareto frontier has shifted with the big-model sweep. "
            f"**Qwen3-Coder-480B (FP8)** lands at MAE {m('L-R01')} (neutral) and {m('L-R04')} (rigorous) "
            "— near or below the human floor — at roughly 5.5 GPU-hours/1k calls (8x A100 80GB, FP8 weights). "
            f"Qwen3-235B-A22B (L-Q01 MAE {m('L-Q01')}) and Qwen2.5-72B (L-N01 MAE {m('L-N01')}) sit on the "
            f"frontier between L-A30M and L-R01. The 30B-A3B MoE (L-A30M MAE {m('L-A30M')}) remains the "
            "cheap workhorse for ~1.2 GPU-hours/1k; the 480B substantially narrows but does not close "
            "the closed-vs-open quality gap."
        )
        # Lower MAE is better. gap > 0 means open-weights (L-R04) beats closed (D01).
        gap = float(d01["MAE_vs_TA_avg"]) - float(lr04["MAE_vs_TA_avg"])
        if gap > 0:
            parts.append(
                f"The headline open-vs-closed gap has *inverted*: best open-weights config (L-R04, MAE "
                f"{m('L-R04')}) is **{gap:.2f} MAE pts better** than the best closed config "
                f"(D01, MAE {m('D01')}). Caveat: cost figures are order-of-magnitude proxies; 480B "
                "inference is much cheaper amortised over many graders than the per-call closed-model API."
            )
        else:
            parts.append(
                f"The headline open-vs-closed gap is small but in favour of the closed model: "
                f"D01 ({m('D01')}) beats best open-weights L-R04 ({m('L-R04')}) by "
                f"**{abs(gap):.2f} MAE pts**."
            )
    else:
        parts.append(
            "On the open-weights side the **Qwen3-30B-A3B MoE** (L-A30M, MAE "
            f"{m('L-A30M')}) is the workhorse — denser models in the family give nothing back."
        )
    L.append(" ".join(parts) + "\n")

    if pair_rows:
        L.append("## Ground truth: pair-level structure and AI behaviour\n")
        L.append(
            f"### Per-pair human disagreement spans {pair_rows[-1]['human_MAE']/pair_rows[0]['human_MAE']:.1f}× "
            f"({pair_rows[0]['human_MAE']:.2f} → {pair_rows[-1]['human_MAE']:.2f} MAE)\n"
        )
        L.append("The 10 fixed grading pairs differ substantially in internal agreement:\n")
        L.append("| pair | n | human MAE | D01 AI MAE | D01 AI bias |")
        L.append("|---|---|---|---|---|")
        for r in pair_rows:
            L.append(f"| {r['pair_label']} | {r['n']} | {r['human_MAE']:.2f} | "
                     f"{r['ai_MAE']:.2f} | {r['ai_bias']:+.2f} |")
        L.append("")
        L.append("### The AI averages out per-pair human noise\n")
        noisiest = max(pair_rows, key=lambda r: r["human_MAE"])
        L.append(
            f"**Pearson r between per-pair human MAE and per-pair D01 MAE = {pearson_r:.3f} "
            f"(p = {pearson_p:.3f}). Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.3f}).** "
            # Judge by significance, not by an |r| cutoff. The cutoff was
            # abs(r) < 0.3, and this study lands at r = -0.301 with p = 0.397
            # over 10 pairs — so it printed "correlated" for a plainly
            # non-significant result, contradicting the paper. Matches the
            # wording rule already used for the headline claim above.
            f"D01's per-pair error is {'uncorrelated' if pearson_p > 0.05 else 'correlated'} with "
            f"the pair's internal disagreement. On the noisiest pair ({noisiest['pair_label']}, "
            f"human MAE {noisiest['human_MAE']:.2f}), AI MAE is {noisiest['ai_MAE']:.2f} — the AI is "
            f"{'*more consistent than the human pair itself*' if noisiest['ai_MAE'] < noisiest['human_MAE'] else 'still less consistent than the human pair'}. "
            "The AI is not inheriting the pair-level noise structure of the ground truth.\n"
        )
        L.append("### Per-TA harshness is real, modest in effect size\n")
        L.append(
            f"The {harsh['n_groups']} TAs span a {harsh['span']:.2f}-pt range in mean awarded total "
            f"score (harshest {harsh['harshest_TA']} at {harsh['harshest_mean']:.2f} / 35, "
            f"most generous {harsh['softest_TA']} at {harsh['softest_mean']:.2f} / 35).\n"
        )
        L.append(
            f"- **One-way ANOVA across {harsh['n_groups']} TAs:** F({harsh['df_between']}, "
            f"{harsh['df_within']}) = **{harsh['F']:.3f}**, p = **{harsh['F_p']:.2g}**"
        )
        L.append(f"- **Kruskal-Wallis:** H = {harsh['H']:.2f}, p = {harsh['H_p']:.2g}")
        L.append(
            f"- **η² = {harsh['eta2']:.3f}** — TA identity explains ≈{harsh['eta2']*100:.0f}% of "
            "variance in awarded scores.\n"
        )
        L.append(
            "The harshness spread is statistically distinguishable from chance but modest in effect "
            "size. Source: [pair_level_ai_mae.md](computer_vision_pair_level_ai_mae.md).\n"
        )

    # Count thinking_mode populations from the tracker for the methods note.
    try:
        _tracker = pd.read_excel(BASE / "computer_vision_results" / "ablation_runs.xlsx", na_filter=False)
        _tm = _tracker["thinking_mode"].value_counts().to_dict() if "thinking_mode" in _tracker.columns else {}
    except Exception:
        _tm = {}
    # The tracker's vocabulary changed with the 2026-08-05 thinking erratum:
    # "max" became "dynamic" (thinking_budget = -1 lets the model pick the
    # budget; it is not a maximum), and D03 gained "default (cannot disable)"
    # because gemini-2.5-pro rejects thinking_budget = 0. Reading the old key
    # silently yielded 0 and regenerated the note claiming zero dynamic runs.
    n_default = _tm.get("default", 0)
    n_max = _tm.get("dynamic", _tm.get("max", 0))
    n_forced = _tm.get("default (cannot disable)", 0)

    L.append("## Method notes and caveats\n")
    L.append(
        f"- Per-question MAE in [master_comparison.md](computer_vision_master_comparison.md) is measured against "
        "the two-TA per-question average (`TA {1,2} - Q{1,2,3} Score`, complete for all 570 "
        "students). It was previously measured against a cross-run median, which drifted as "
        "unrelated runs were added; those older per-question numbers are not comparable to these.\n"
        "- Question 4 is a *bonus* question (max 5 bonus pts); the headline MAE-vs-TA-avg metric is "
        "computed on the 35-pt base total, not base + bonus.\n"
        f"- G-series and M-series Gemini runs are on Number ≤ 100 only (3-flash-preview throughput). "
        "All reported G/M comparisons against D01 use the matched D01 ≤ 100 subset.\n"
        "- TA-average target uses both TAs when both are present, otherwise the single available "
        "TA's score; we do not impute zeros for missing TAs.\n"
        "- Slot labels (`TA_1_ID` / `TA_2_ID`) are slot positions, not fixed identities. Signed "
        "slot1 − slot2 differences are only meaningful within a pair.\n"
        # Corrected 2026-08-05: this paragraph used to claim the default runs
        # used "the model's default thinking policy". They did not — the
        # grading script passes thinking_budget = 0 for Gemini-3.x IDs, i.e.
        # thinking OFF (verified against computer_vision_grade_with_gemini.py's
        # thinking logic, the tracker's thinking_mode/command columns, and the
        # run logs, which print "Reasoning/thinking is DISABLED"). The old
        # wording made B03-vs-A01 read as max-vs-default thinking when it is
        # actually dynamic-vs-off, and regenerating this report used to
        # overwrite the corrected text with the wrong claim.
        f"- **Gemini thinking configuration.** {n_default} of the Gemini ablations were run with "
        "thinking explicitly disabled (`thinking_budget = 0`, the grading script's default for "
        f"Gemini-3.x model IDs); {n_max} ablations (B03, F01, G03) were run with dynamic thinking "
        "enabled (`thinking_budget = -1`; the model selects the per-call budget); D03 "
        "(`gemini-2.5-pro`) rejects `thinking_budget = 0` and ran under its default dynamic policy. "
        "The `thinking_mode` column in `computer_vision_results/ablation_runs.xlsx` records the "
        "per-run setting. The B03 vs A01 comparison therefore measures dynamic-budget thinking "
        "against thinking-off on Flash-Lite. "
        "*(Corrected 2026-08-05 against `computer_vision_grade_with_gemini.py` lines 502-535 and "
        "the run tracker; an earlier version of this note said the 20 ran under the API's default "
        "thinking policy and mislabelled `-1` as maximum thinking.)*\n"
    )

    L.append("## Deliverables\n")
    L.append("- [master_comparison.csv](computer_vision_master_comparison.csv) / [.md](computer_vision_master_comparison.md) — full "
             f"{len(master_df)}-row table with bootstrap CIs.")
    L.append("- [human_floor.md](computer_vision_human_floor.md) — inter-grader stats, per-pair breakdown, per-TA harshness.")
    L.append("- [n100_restricted_comparison.md](computer_vision_n100_restricted_comparison.md) — D01 vs G/M on matched n = 100.")
    L.append("- [pair_level_ai_mae.md](computer_vision_pair_level_ai_mae.md) — pair-correlation analysis + F-test.")
    L.append("- [variance.md](computer_vision_variance.md) — E-series temperature stability.")
    L.append("- [failure_cases.md](computer_vision_failure_cases.md) — D01 vignettes + strict-collapse mechanism evidence.")
    L.append("- [../media/](../media/) — paper figures (PDF, 300 DPI, fonts embedded).")

    (ANALYSIS / "computer_vision_new_summary.md").write_text("\n".join(L), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print(">> Discovering ablations…")
    ablations = list_ablations()
    by_source = {}
    for a in ablations:
        by_source[a["source"]] = by_source.get(a["source"], 0) + 1
    print(f"   {len(ablations)} ablations found ("
          + " + ".join(f"{n} {s}" for s, n in sorted(by_source.items())) + ")")

    print(">> Building master comparison table…")
    master_df, cached = build_master_table(ablations)
    master_df.to_csv(ANALYSIS / "computer_vision_master_comparison.csv", index=False)
    print(f"   wrote {ANALYSIS / 'computer_vision_master_comparison.csv'}")

    print(">> Computing human floor…")
    floor = human_floor()
    write_human_floor_md(floor)
    print(f"   TA1 vs TA2: MAE={floor['MAE']:.2f}, Pearson r={floor['pearson']:.3f}, "
          f"max={floor['max_disagreement']:.1f}, n={floor['n_pairs']}")

    write_master_md(master_df, floor)

    print(">> Variance analysis…")
    v_df, per_cell_panels = variance_analysis()
    write_variance_md(v_df)
    print(v_df.to_string())

    print(">> Failure cases…")
    rid, over, under = failure_cases(master_df, cached)
    write_failure_cases_md(rid, over, under, cached)
    print(f"   best config: {rid}")

    print(">> Plots…")
    make_plots(master_df, cached, per_cell_panels)


    print(">> Pair-level + per-TA stats…")
    master_grades = pd.read_excel(BASE / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx")
    write_pair_level_ai_mae_md(master_grades, cached, master_df)

    print(">> n100 restricted comparison…")
    write_n100_restricted_md(master_df, cached)

    print(">> new_summary.md (paper-style)…")
    write_new_summary_md(master_df, floor, v_df, cached)

    print("DONE.")


if __name__ == "__main__":
    main()

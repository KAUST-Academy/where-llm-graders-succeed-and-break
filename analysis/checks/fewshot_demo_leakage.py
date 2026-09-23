#!/usr/bin/env python3
"""
Few-shot leakage: recompute the K-series few-shot runs on a strict
hold-out that excludes the demonstration students.

Question: the five CV demonstration students
remain in the evaluated cohort of K01 / L-K01 and are graded with their own
worked answer in the prompt. Recompute MAE / bias / 95% CI for K01 and L-K01
with those students removed, against A01 and L-A30M restricted to the same
students, and report the few-shot uplift with and without them.

Conventions (task rules, identical to the paper's analysis scripts):
  * CV AI total   = workbook column "AI Total Score" (Q1-Q3 base, 35 points).
  * CV TA truth   = mean of "TA 1 - Total Score (out of 35)" and
                    "TA 2 - Total Score (out of 35)".
  * Rows with a blank AI total or a blank TA total are excluded.
  * MAE = mean |AI - TA_avg|; bias = mean (AI - TA_avg).
  * 95% CI = percentile bootstrap over students, 2000 resamples,
    numpy.random.default_rng(seed=0) -- the same routine as
    analysis/computer_vision_run_analysis.py::bootstrap_mae_ci, so the
    full-cohort rows reproduce Table 19 (tab:allruns) exactly.
  * Human floor = mean |TA1 - TA2| over the same students.
  * ML exam (extension only): AI total = "AI Total Score" + "AI Total Bonus",
    TA truth = mean of "TA 1 - Total Grade" / "TA 2 - Total Grade" from
    introduction_to_ai_dataset/Practical_AI_exam_grades.csv.

Run:   python3 analysis/checks/fewshot_demo_leakage.py
Output is Markdown; the saved copy is analysis/checks/fewshot_demo_leakage.md.
No GPU, no API calls, no tracked file modified.
"""
from __future__ import annotations

import importlib.util
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = Path(__file__).resolve().parents[2]
CV_RES = ROOT / "computer_vision_results" / "results"
CV_MASTER = ROOT / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"
CV_BANK = ROOT / "computer_vision_dataset" / "few_shot_examples.json"
CV_BUILDER = ROOT / "computer_vision_build_few_shot_examples.py"
ML_RES = ROOT / "introduction_to_ai_results" / "results"
ML_MASTER = ROOT / "introduction_to_ai_dataset" / "Practical_AI_exam_grades.csv"
ML_BANK = ROOT / "introduction_to_ai_dataset" / "few_shot_examples.json"

N_BOOT = 2000
SEED = 0

CV_RUNS = {
    "A01": "A01__A01_m-flash-lite_sol1_gd1_rs0_bd1_str-neutral_t00_n1.xlsx",
    "K01": "K01__K01_m-flash-lite_sol1_gd1_rs0_bd1_str-neutral_t00_n1_fs2.xlsx",
    "L-A30M": "L-A30M__L-A30M_m-qwen3-coder-30b-a3b_sol1_gd1_rs0_bd1_str-neutral_t00_n1.xlsx",
    "L-K01": "L-K01__L-K01_m-qwen3-coder-30b-a3b_sol1_gd1_rs0_bd1_str-neutral_t00_n1_fs2.xlsx",
}
# Paper values for comparison: Table 11 (tab:fewshot) MAE/bias; Table 19
# (tab:allruns) n and 95% CI.  (n, MAE, (lo, hi), bias)
PAPER_CV = {
    "A01": (570, 3.34, (3.15, 3.53), -1.78),
    "K01": (567, 3.03, (2.83, 3.24), +1.09),
    "L-A30M": (570, 4.50, (4.29, 4.72), -2.70),
    "L-K01": (570, 3.34, (3.16, 3.53), -1.77),
}
CV_PAIRS = [("Flash-Lite", "A01", "K01"), ("Qwen3-Coder-30B-A3B", "L-A30M", "L-K01")]

ML_RUNS = {
    "IA01-Q30": "IA01__IA01_m-qwen3-coder-30b-a3b_sol1_gd0_rs0_bd1_str-neutral_t00_n1.xlsx",
    "IA201": "IA201__IA201_m-qwen3-coder-30b-a3b_sol1_gd0_rs0_bd1_str-neutral_t00_n1_fs2.xlsx",
    "IG01": "IG01__IG01_m-flash-lite_sol1_gd0_rs0_bd1_str-neutral_t00_n1.xlsx",
    "IG18": "IG18__IG18_m-flash-lite_sol1_gd0_rs0_bd1_str-neutral_t00_n1.xlsx",
}
ML_PAIRS = [("Qwen3-Coder-30B-A3B", "IA01-Q30", "IA201"), ("Flash-Lite", "IG01", "IG18")]


# --------------------------------------------------------------------------- helpers
def f3(x: float) -> str:
    return f"{x:.3f}"


def s3(x: float) -> str:
    return f"{x:+.3f}"


def ci3(c: tuple[float, float]) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def md_table(header: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def _boot_idx(n: int) -> np.ndarray:
    """One fresh default_rng(SEED) per statistic, as in the paper's bootstrap_mae_ci."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def _pct(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def ci_mae(diffs) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=float)
    return _pct(np.abs(d[_boot_idx(len(d))]).mean(axis=1))


def ci_mean(vals) -> tuple[float, float]:
    d = np.asarray(vals, dtype=float)
    return _pct(d[_boot_idx(len(d))].mean(axis=1))


def ci_paired_delta_mae(d_base, d_few) -> tuple[float, float]:
    """Paired bootstrap of MAE(base) - MAE(few): students resampled jointly."""
    a = np.abs(np.asarray(d_base, dtype=float))
    b = np.abs(np.asarray(d_few, dtype=float))
    assert len(a) == len(b)
    idx = _boot_idx(len(a))
    return _pct((a[idx] - b[idx]).mean(axis=1))


def metrics(diffs) -> dict:
    d = np.asarray(diffs, dtype=float)
    return {"n": int(len(d)), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "mae_ci": ci_mae(d), "bias_ci": ci_mean(d)}


def load_cv(rid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-student frame (index = Number) with ai, t1, t2, ta, diff, valid; plus the raw sheet."""
    path = CV_RES / CV_RUNS[rid]
    raw = pd.read_excel(path)
    ai = pd.to_numeric(raw["AI Total Score"], errors="coerce")
    t1 = pd.to_numeric(raw["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(raw["TA 2 - Total Score (out of 35)"], errors="coerce")
    # Cross-check the stored TA totals against the Q1-Q3 rebuild the paper's
    # analysis script uses (ta_combined); a mismatch would mean the stored
    # formula cells were dropped on save.
    r1 = sum(pd.to_numeric(raw[f"TA 1 - Q{q} Score"], errors="coerce") for q in (1, 2, 3))
    r2 = sum(pd.to_numeric(raw[f"TA 2 - Q{q} Score"], errors="coerce") for q in (1, 2, 3))
    mism = int(((r1 - t1).abs() > 1e-9).sum() + ((r2 - t2).abs() > 1e-9).sum())
    out = pd.DataFrame({"ai": ai.values, "t1": t1.values, "t2": t2.values},
                       index=raw["Number"].astype(int).values)
    out.index.name = "Number"
    out["ta"] = (out.t1 + out.t2) / 2.0
    out["diff"] = out.ai - out.ta
    out["valid"] = out.ai.notna() & out.t1.notna() & out.t2.notna()
    blank_ai = out.index[out.ai.isna()].tolist()
    blank_ta = out.index[out.t1.isna() | out.t2.isna()].tolist()
    print(f"- `{path.relative_to(ROOT)}`: {len(out)} rows; blank AI total: {len(blank_ai)} "
          f"{blank_ai if blank_ai else ''}; blank TA total: {len(blank_ta)} "
          f"{blank_ta if blank_ta else ''}; valid rows: {int(out.valid.sum())}; "
          f"stored-vs-rebuilt TA total mismatches: {mism}")
    return out, raw.set_index(raw["Number"].astype(int))


def pair_block(label: str, base_id: str, few_id: str, base: pd.DataFrame, few: pd.DataFrame,
               demo: list[int], extra_sets: dict[str, list[int]] | None = None) -> list[dict]:
    """Base vs few-shot on the same students, with and without the demo students."""
    common = sorted(set(base.index[base.valid]) & set(few.index[few.valid]))
    holdout = [n for n in common if n not in demo]
    demo_in = [n for n in demo if n in common]
    sets = {"with demonstration students": common,
            "without demonstration students (strict hold-out)": holdout}
    if extra_sets:
        sets.update(extra_sets)
    print(f"**{label}: {base_id} (no few-shot) vs {few_id} (+ few-shot)** -- students valid in "
          f"both runs: {len(common)}; demonstration students among them: {demo_in} "
          f"({len(demo_in)}); strict hold-out: {len(holdout)} students.\n")
    rows, recs = [], []
    for sname, S in sets.items():
        db = base.loc[S, "diff"].values
        df_ = few.loc[S, "diff"].values
        mb, mf = metrics(db), metrics(df_)
        delta = mb["mae"] - mf["mae"]
        dci = ci_paired_delta_mae(db, df_)
        dbias = mf["bias"] - mb["bias"]
        rec = {"set": sname, "n": mb["n"], "base": mb, "few": mf, "delta_mae": delta,
               "delta_ci": dci, "delta_bias": dbias}
        recs.append(rec)
        rows.append([sname, mb["n"],
                     f"{f3(mb['mae'])} {ci3(mb['mae_ci'])}", s3(mb["bias"]),
                     f"{f3(mf['mae'])} {ci3(mf['mae_ci'])}", s3(mf["bias"]),
                     f"{s3(delta)} {ci3(dci)}", s3(dbias)])
    md_table(["students", "n", f"{base_id} MAE [95% CI]", f"{base_id} bias",
              f"{few_id} MAE [95% CI]", f"{few_id} bias",
              "uplift = MAE(base) - MAE(few) [paired 95% CI]", "bias shift (few - base)"], rows)
    # bias CIs, kept out of the main table for width
    rows = []
    for rec in recs:
        rows.append([rec["set"], rec["n"], ci3(rec["base"]["bias_ci"]), ci3(rec["few"]["bias_ci"])])
    md_table(["students", "n", f"{base_id} bias 95% CI", f"{few_id} bias 95% CI"], rows)
    # arithmetic decomposition for the few-shot run
    if demo_in:
        n_with = len(common)
        s_demo = float(np.abs(few.loc[demo_in, "diff"]).sum())
        mae_with = float(np.abs(few.loc[common, "diff"]).mean())
        mae_wo = (n_with * mae_with - s_demo) / (n_with - len(demo_in))
        print(f"Check for {few_id}: MAE_without = (n*MAE_with - sum_demo|err|)/(n-{len(demo_in)}) = "
              f"({n_with}*{mae_with:.6f} - {s_demo:.4f})/{n_with - len(demo_in)} = {mae_wo:.6f}; "
              f"mean |err| of {few_id} on the {len(demo_in)} demonstration students = "
              f"{s_demo / len(demo_in):.3f} vs {mae_with:.3f} on the full common set.\n")
    return recs


# --------------------------------------------------------------------------- main
def main() -> None:
    print("# Few-shot leakage: K01 / L-K01 on a strict hold-out\n")
    print(f"Bootstrap: percentile, {N_BOOT} resamples over students, numpy default_rng(seed={SEED}), "
          "one fresh generator per statistic (so two statistics on the same n share the same "
          "resample indices). Paired uplift CIs resample students jointly for the base and "
          "few-shot run.\n")

    # ------------------------------------------------------------------ 0. ids
    print("## 0. The demonstration students\n")
    bank = json.loads(CV_BANK.read_text(encoding="utf-8"))
    rows = []
    json_per_q: dict[int, list[int]] = {}
    for q, examples in bank.items():
        qn = int(q[1:])
        json_per_q[qn] = [int(e["provenance"]["student_number"]) for e in examples]
        for e in examples:
            p = e["provenance"]
            rows.append([q, p["student_number"], "/".join(p["ta_pair"]), p["ta_avg_total"],
                         p["score_source"], e["score"], e["bonus"]])
    md_table(["question", "student", "TA pair (provenance)", "TA-avg total (provenance)",
              "label source", "demo score", "demo bonus"], rows)
    DEMO = sorted({n for v in json_per_q.values() for n in v})
    print(f"Union of demonstration students from `{CV_BANK.relative_to(ROOT)}` "
          f"(field `provenance.student_number`): **{DEMO}** ({len(DEMO)} students; "
          f"{sum(len(v) for v in json_per_q.values())} examples, 2 per question x 4 questions; "
          f"336 appears in Q1, Q2 and Q4, 340 in Q1 and Q2).\n")

    print("### 0a. Re-derivation from `computer_vision_build_few_shot_examples.py`\n")
    try:
        spec = importlib.util.spec_from_file_location("cv_builder", CV_BUILDER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # defines functions only; main() is under __main__
        pool = mod.candidate_pool()
        d01 = pd.read_excel(mod.D01_XLSX).set_index("Number")
        rederived = {q: [int(n) for n in mod.pick_for_question(pool, q, d01, mod.K_PER_QUESTION)]
                     for q in (1, 2, 3, 4)}
        print(f"- candidate pool (`{mod.HIGH_AGREEMENT_PAIR}`, |TA1-TA2| <= {mod.MAX_AGREEMENT_GAP}, "
              f"TA-avg > 0): {len(pool)} students from `{Path(mod.MASTER).relative_to(ROOT)}`")
        print(f"- D01 labels from `{Path(mod.D01_XLSX).relative_to(ROOT)}`; notebooks checked under "
              f"`{Path(mod.EXTRACTED).relative_to(ROOT)}`")
        ok = all(rederived[q] == json_per_q[q] for q in (1, 2, 3, 4))
        for q in (1, 2, 3, 4):
            print(f"- Q{q}: builder picks {rederived[q]}; JSON has {json_per_q[q]} -> "
                  f"{'match' if rederived[q] == json_per_q[q] else 'MISMATCH'}")
        print(f"- re-derived union: {sorted({n for v in rederived.values() for n in v})} -> "
              f"{'identical to the JSON' if ok else 'DIFFERS from the JSON (JSON is what the runs used)'}\n")
    except Exception as exc:  # the JSON provenance remains authoritative
        print(f"- re-derivation failed ({type(exc).__name__}: {exc}); the graders read the JSON, "
              "so its provenance field is what the K-series runs actually used.\n")
        traceback.print_exc()

    print("### 0b. Cross-check against the master sheet\n")
    master = pd.read_excel(CV_MASTER)
    m = master[master["Number"].isin(DEMO)].copy()
    m["t1"] = pd.to_numeric(m["TA 1 - Total Score (out of 35)"], errors="coerce")
    m["t2"] = pd.to_numeric(m["TA 2 - Total Score (out of 35)"], errors="coerce")
    rows = [[int(r.Number), r.TA_1_ID, r.TA_2_ID, r.t1, r.t2, (r.t1 + r.t2) / 2, abs(r.t1 - r.t2)]
            for r in m.itertuples()]
    md_table(["student", "TA_1_ID", "TA_2_ID", "TA1 total", "TA2 total", "TA avg", "|TA1-TA2|"], rows)
    pair_ok = all((a, b) in {("TA_6", "TA_16"), ("TA_16", "TA_6")} for a, b in zip(m.TA_1_ID, m.TA_2_ID))
    print(f"`{CV_MASTER.relative_to(ROOT)}` rows for the five ids: all graded by the TA_6/TA_16 pair: "
          f"{pair_ok}; max |TA1-TA2| = {float((m.t1 - m.t2).abs().max()):.3f} (README claim: "
          "highest-agreement pair TA_6/TA_16, gap <= 1).\n")

    # ------------------------------------------------------------------ 1. load CV runs
    print("## 1. CV run workbooks (`computer_vision_results/results/`)\n")
    cv, cv_raw = {}, {}
    for rid in CV_RUNS:
        cv[rid], cv_raw[rid] = load_cv(rid)
    print()
    print("K01's three blank rows (65, 326, 379) each lack one of the Q1-Q3 base scores in the "
          "workbook, so they are not recoverable and are excluded, giving n = 567 as in Table 19.\n")

    # ------------------------------------------------------------------ 2. full cohort
    print("## 2. Full-cohort reproduction of Table 11 / Table 19 (each run on its own valid rows)\n")
    rows = []
    for rid in CV_RUNS:
        d = cv[rid]
        mt = metrics(d.loc[d.valid, "diff"].values)
        pn, pm, pci, pb = PAPER_CV[rid]
        rows.append([rid, mt["n"], f3(mt["mae"]), ci3(mt["mae_ci"]), s3(mt["bias"]), ci3(mt["bias_ci"]),
                     f"n={pn}, MAE {pm:.2f} {list(pci)}, bias {pb:+.2f}",
                     "yes" if (mt["n"] == pn and round(mt["mae"], 2) == pm and round(mt["bias"], 2) == pb
                               and round(mt["mae_ci"][0], 2) == pci[0] and round(mt["mae_ci"][1], 2) == pci[1])
                     else "NO"])
    md_table(["run", "n", "MAE", "MAE 95% CI", "bias", "bias 95% CI", "paper (Tables 11/19)",
              "reproduces at 2 d.p."], rows)

    # ------------------------------------------------------------------ 3. hold-out
    print("## 3. Strict hold-out: base vs few-shot on the same students, with and without the five\n")
    recs = {}
    for label, b, f in CV_PAIRS:
        recs[(b, f)] = pair_block(label, b, f, cv[b], cv[f], DEMO)

    print("### 3a. All four runs on one common student set\n")
    common4 = sorted(set.intersection(*[set(cv[r].index[cv[r].valid]) for r in CV_RUNS]))
    hold4 = [n for n in common4 if n not in DEMO]
    rows = []
    for sname, S in (("with demonstration students", common4),
                     ("without demonstration students (strict hold-out)", hold4)):
        for rid in CV_RUNS:
            mt = metrics(cv[rid].loc[S, "diff"].values)
            rows.append([sname, rid, mt["n"], f3(mt["mae"]), ci3(mt["mae_ci"]), s3(mt["bias"]),
                         ci3(mt["bias_ci"])])
    md_table(["students", "run", "n", "MAE", "MAE 95% CI", "bias", "bias 95% CI"], rows)
    print(f"Common set = students valid in all four runs: {len(common4)}; hold-out: {len(hold4)}.\n")

    # ------------------------------------------------------------------ 4. the five students
    print("## 4. The five demonstration students under each run\n")
    rows = []
    for n in DEMO:
        r = [n, cv["A01"].loc[n, "t1"], cv["A01"].loc[n, "t2"], cv["A01"].loc[n, "ta"]]
        for rid in CV_RUNS:
            r += [cv[rid].loc[n, "ai"], s3(cv[rid].loc[n, "diff"])]
        rows.append(r)
    md_table(["student", "TA1", "TA2", "TA avg", "A01 total", "A01 err", "K01 total", "K01 err",
              "L-A30M total", "L-A30M err", "L-K01 total", "L-K01 err"], rows)
    for rid in CV_RUNS:
        e = np.abs(cv[rid].loc[DEMO, "diff"].values)
        print(f"- {rid} on the five: mean |err| = {e.mean():.3f}, bias = "
              f"{cv[rid].loc[DEMO, 'diff'].mean():+.3f} (n = 5)")
    print()

    print("### 4a. Did the few-shot runs copy the in-context label on the demonstrated question?\n")
    print("For every (student, question) that appears in the example bank: the demonstration's "
          "label (D01's score), what K01 and L-K01 awarded that student on that question, and the "
          "TA per-question scores (Q4 is bonus-only and has no TA score column).\n")
    rows = []
    copies = {"K01": [0, 0], "L-K01": [0, 0]}
    for q in (1, 2, 3, 4):
        for e in bank[f"Q{q}"]:
            n = int(e["provenance"]["student_number"])
            lab = float(e["score"])
            k = float(cv_raw["K01"].loc[n, f"AI Q{q} Score"])
            lk = float(cv_raw["L-K01"].loc[n, f"AI Q{q} Score"])
            a = float(cv_raw["A01"].loc[n, f"AI Q{q} Score"])
            la = float(cv_raw["L-A30M"].loc[n, f"AI Q{q} Score"])
            if q <= 3:
                ta1 = float(cv_raw["K01"].loc[n, f"TA 1 - Q{q} Score"])
                ta2 = float(cv_raw["K01"].loc[n, f"TA 2 - Q{q} Score"])
                ta = f"{ta1:g} / {ta2:g}"
            else:
                ta = "n/a"
            for rid, v in (("K01", k), ("L-K01", lk)):
                copies[rid][1] += 1
                copies[rid][0] += int(abs(v - lab) < 1e-9)
            rows.append([f"Q{q}", n, f"{lab:g}", f"{a:g}", f"{k:g}", f"{la:g}", f"{lk:g}", ta])
    md_table(["question", "student", "demo label (D01)", "A01", "K01", "L-A30M", "L-K01",
              "TA1 / TA2"], rows)
    for rid, (c, t) in copies.items():
        print(f"- {rid} reproduced the demonstration label exactly on {c} of {t} demonstrated "
              "(student, question) cells")
    print()

    # ------------------------------------------------------------------ 5. human floor
    print("## 5. Human floor (mean |TA1 - TA2|) on each student set\n")
    all570 = sorted(cv["A01"].index[cv["A01"].valid])
    sets = {
        "all 570": all570,
        "570 minus the five": [n for n in all570 if n not in DEMO],
        "Flash-Lite common set (567)": sorted(set(cv["A01"].index[cv["A01"].valid]) & set(cv["K01"].index[cv["K01"].valid])),
    }
    sets["Flash-Lite hold-out (562)"] = [n for n in sets["Flash-Lite common set (567)"] if n not in DEMO]
    rows = []
    for sname, S in sets.items():
        g = (cv["A01"].loc[S, "t1"] - cv["A01"].loc[S, "t2"]).values
        rows.append([sname, len(S), f3(np.abs(g).mean()), ci3(ci_mae(g))])
    md_table(["students", "n", "human floor", "95% CI"], rows)

    # ------------------------------------------------------------------ 6. ML extension
    print("## 6. Extension: the ML exam's few-shot arms, same recompute\n")
    print("Appendix I says the ML demonstrations carry three students who remain in the cohort. "
          "The CV K-series is the primary check; this section extends it to the ML exam.\n")
    ml_bank = json.loads(ML_BANK.read_text(encoding="utf-8"))
    ml_per_q = {q: [int(e["provenance"]["student_number"]) for e in ex] for q, ex in ml_bank.items()}
    ML_DEMO = sorted({n for v in ml_per_q.values() for n in v})
    print(f"- `{ML_BANK.relative_to(ROOT)}`: per question {ml_per_q}; union **{ML_DEMO}**")
    mlm = pd.read_csv(ML_MASTER)
    mt1 = pd.to_numeric(mlm["TA 1 - Total Grade"], errors="coerce")
    mt2 = pd.to_numeric(mlm["TA 2 - Total Grade"], errors="coerce")
    ta_ml = pd.DataFrame({"t1": mt1.values, "t2": mt2.values}, index=mlm["Number"].astype(int).values)
    ta_ml["ta"] = (ta_ml.t1 + ta_ml.t2) / 2.0
    sub = mlm[mlm["Number"].isin(ML_DEMO)]
    print(f"- `{ML_MASTER.relative_to(ROOT)}` rows: " + "; ".join(
        f"{int(r.Number)}: {r.TA_1_ID}/{r.TA_2_ID}, TA totals {r._4:g}/{r._5:g}"
        for r in sub[["Number", "TA_1_ID", "TA_2_ID", "TA 1 - Total Grade", "TA 2 - Total Grade"]].itertuples())
          + f"; blank TA totals in the CSV: {int((mt1.isna() | mt2.isna()).sum())}\n")

    def load_ml(rid: str) -> pd.DataFrame:
        path = ML_RES / ML_RUNS[rid]
        raw = pd.read_excel(path)
        score = pd.to_numeric(raw["AI Total Score"], errors="coerce")
        bonus = pd.to_numeric(raw["AI Total Bonus"], errors="coerce").fillna(0.0)
        out = pd.DataFrame({"ai": (score + bonus).values}, index=raw["Number"].astype(int).values)
        out = out.join(ta_ml, how="left")
        out["diff"] = out.ai - out.ta
        out["valid"] = score.notna().values & out.t1.notna() & out.t2.notna()
        # cross-check the workbook's own TA columns against the CSV
        w1 = pd.to_numeric(raw["TA 1 - Total Grade"], errors="coerce").values
        w2 = pd.to_numeric(raw["TA 2 - Total Grade"], errors="coerce").values
        mism = int((np.abs(w1 - out.t1.values) > 1e-9).sum() + (np.abs(w2 - out.t2.values) > 1e-9).sum())
        print(f"- `{path.relative_to(ROOT)}`: {len(out)} rows; blank AI score: {int(score.isna().sum())}; "
              f"valid rows: {int(out.valid.sum())}; workbook-vs-CSV TA total mismatches: {mism}")
        return out

    ml = {rid: load_ml(rid) for rid in ML_RUNS}
    print()
    print("Paper values (Appendix G): IA01 11.12 / +10.44 -> IA201 5.84 / +4.56 on the full cohort; "
          "on the 679-student matched subsample IG01 7.87 / +6.52 -> IG18 4.32 / +0.37. Table 20's CIs "
          "use 50,000 resamples; the CIs below use the task rule (2000, seed 0).\n")
    for label, b, f in ML_PAIRS:
        pair_block(label, b, f, ml[b], ml[f], ML_DEMO)

    # ------------------------------------------------------------------ 7. summary
    print("## 7. Summary\n")
    for (b, f), rr in recs.items():
        w, wo = rr[0], rr[1]
        print(f"- {b} -> {f}: with the five (n={w['n']}): {f3(w['base']['mae'])} -> {f3(w['few']['mae'])}, "
              f"uplift {s3(w['delta_mae'])} {ci3(w['delta_ci'])}, bias {s3(w['base']['bias'])} -> "
              f"{s3(w['few']['bias'])}; strict hold-out (n={wo['n']}): {f3(wo['base']['mae'])} -> "
              f"{f3(wo['few']['mae'])}, uplift {s3(wo['delta_mae'])} {ci3(wo['delta_ci'])}, bias "
              f"{s3(wo['base']['bias'])} -> {s3(wo['few']['bias'])}; {f} MAE CI on the hold-out "
              f"{ci3(wo['few']['mae_ci'])}, change in {f} MAE from dropping the five: "
              f"{s3(wo['few']['mae'] - w['few']['mae'])}")
    print()
    print("Six-decimal values for the strict hold-out rows (for two-decimal rounding in the paper):\n")
    rows = []
    for (b, f), rr in recs.items():
        wo = rr[1]
        for rid, mt in ((b, wo["base"]), (f, wo["few"])):
            rows.append([rid, wo["n"], f"{mt['mae']:.6f}", f"{mt['mae_ci'][0]:.6f}", f"{mt['mae_ci'][1]:.6f}",
                         f"{mt['bias']:+.6f}"])
        rows.append([f"uplift {b} -> {f}", wo["n"], f"{wo['delta_mae']:+.6f}", f"{wo['delta_ci'][0]:.6f}",
                     f"{wo['delta_ci'][1]:.6f}", f"bias shift {wo['delta_bias']:+.6f}"])
    md_table(["run", "n", "MAE", "MAE CI lo", "MAE CI hi", "bias"], rows)


if __name__ == "__main__":
    main()

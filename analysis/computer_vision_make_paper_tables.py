"""Emit the LaTeX bodies for the Section 5 tables, and verify them.

Why a generator: Table 1 went from 21 hand-typed rows (Qwen + Gemini) to 45+
across seven families. Hand-maintaining that through a pipeline change is how
a paper ends up quoting a number no CSV contains. Everything here is derived
from analysis/computer_vision_master_comparison.csv and the result workbooks; nothing is typed.

The team's convention is hand-written LaTeX in sections/*.tex, so this prints to
stdout for pasting rather than writing \\input files. Rerun after any pipeline
change and diff against the .tex to catch drift:

    python analysis/computer_vision_make_paper_tables.py            # emit the tables
    python analysis/computer_vision_make_paper_tables.py --verify   # re-derive every number in
                                                    # sections/brittleness.tex

Run on a compute node -- --verify opens the result workbooks.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import computer_vision_run_analysis as ra  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CSV = Path(__file__).parent / "computer_vision_master_comparison.csv"
WORDINGS = ["strict", "rigorous", "exacting"]

# Display names: the CSV carries lowercase pipeline tags; the paper uses vendor
# capitalisation. Keyed on the tag so a new model fails loudly rather than
# silently printing a tag where a name belongs.
DISPLAY = {
    "qwen2.5-coder-7b": "Qwen2.5-Coder-7B", "qwen2.5-coder-14b": "Qwen2.5-Coder-14B",
    "qwen2.5-coder-32b": "Qwen2.5-Coder-32B", "qwen2.5-72b": "Qwen2.5-72B",
    "qwen3-coder-30b-a3b": "Qwen3-Coder-30B-A3B", "qwen3-coder-next": "Qwen3-Coder-Next 80B",
    "qwen3-235b-a22b": "Qwen3-235B-A22B", "qwen3-coder-480b": "Qwen3-Coder-480B",
    "llama-3.1-8b": "Llama-3.1-8B", "llama-3.3-70b": "Llama-3.3-70B",
    "glm-4-9b": "GLM-4-9B", "glm-4-32b": "GLM-4-32B", "glm-4.5-air": "GLM-4.5-Air",
    "gemma-3-12b": "Gemma-3-12B", "gemma-3-27b": "Gemma-3-27B",
    "mistral-small-24b": "Mistral-Small-24B",
    "deepseek-coder-v2-lite": "DeepSeek-Coder-V2-Lite",
    "flash-lite": "Flash-Lite", "3-flash-preview": "3-flash-preview",
    "3.1-pro-preview": "3.1-pro-preview", "2.5-pro": "2.5-pro",
    "gpt-5.5": "GPT-5.5", "gpt-5.4": "GPT-5.4", "claude-opus-5": "Claude Opus 5",
}
BEHAVIOUR_TEX = {"graded": "graded", "collapse": "collapse",
                 "refusal": "\\textbf{refusal}", "near-refusal": "near-refusal"}


def load():
    d = pd.read_csv(CSV)
    d["family"] = [ra.model_family(m, s) for m, s in zip(d.model, d.source)]
    d["params"] = [ra.model_params_b(m) for m in d.model]
    return d


def default_config(d, min_n=ra.FULL_COHORT_MIN_N):
    sel = d[d.n_valid_vs_TA >= min_n].copy()
    for k, v in ra.DEFAULT_PROMPT_CONFIG.items():
        if k in sel.columns:
            sel = sel[sel[k] == v]
    return sel


def name(tag):
    if tag not in DISPLAY:
        raise KeyError(f"no display name for model tag {tag!r} -- add it to DISPLAY")
    return DISPLAY[tag]


# ---------------------------------------------------------------------------
# Table 1: the strict ladder -- one row per model, neutral baseline alongside.
# Ordered by parameter count so the size claim is readable off the table
# instead of asserted in prose, and grouped only by open/closed.
# ---------------------------------------------------------------------------
def table_ladder(d):
    pairs = ra.matched_persona_pairs(d)
    op = pairs[~pairs.family.isin(ra.CLOSED_FAMILIES)].sort_values(["params_b", "model"])
    gem = pairs[pairs.family == "Gemini"]
    out = []
    out.append(r"\multicolumn{8}{l}{\emph{Open weights --- ordered by total parameters}} \\")
    for r in op.itertuples():
        sz = f"{r.params_b:.0f}B" if pd.notna(r.params_b) else "---"
        out.append(
            f"{r.test_rid} & {name(r.model)} & {r.family} & {sz} & "
            f"{r.baseline_MAE:.2f} & \\textbf{{{r.test_MAE:.2f}}} & "
            f"$\\times {r.ratio:.2f}$ & {BEHAVIOUR_TEX[r.test_behaviour]} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{8}{l}{\emph{Closed --- reference}} \\")
    for r in gem.itertuples():
        out.append(
            f"{r.test_rid} & {name(r.model)} & {r.family} & --- & "
            f"{r.baseline_MAE:.2f} & {r.test_MAE:.2f} & "
            f"$\\times {r.ratio:.2f}$ & {BEHAVIOUR_TEX[r.test_behaviour]} \\\\")
    return "\n".join(out), pairs


# ---------------------------------------------------------------------------
# Table 2: wording. Same model, three strict-flavoured presets. The presets are
# NOT minimal pairs -- only `strict` carries the two policy sentences -- so this
# table is evidence about the policy text, not about the adjective. That is
# stated in the caption, because the obvious misreading is lexical.
# ---------------------------------------------------------------------------
def table_wording(d):
    sel = default_config(d)
    # Take the neutral baseline from matched_persona_pairs, not from a groupby:
    # flash-lite has TWO neutral runs at the default config (A01 n=570 single-run,
    # E01 n=549 averaged over 5), and picking whichever sorts first would print
    # 3.38 here against 3.34 in Table 1 -- the same number, two values, one paper.
    baseline = {r.model: r.baseline_MAE
                for r in ra.matched_persona_pairs(d).itertuples()}
    rows = []
    for model, g in sel.groupby("model"):
        v = {r.strictness: r for r in g.itertuples()}
        if not all(w in v for w in WORDINGS):
            continue
        rows.append((ra.model_family(model, g.iloc[0]["source"]),
                     ra.model_params_b(model), model, baseline.get(model), v))
    rows.sort(key=lambda r: (r[0] != "Gemini", r[1] if r[1] is not None else 1e9))
    out = []
    for fam, params, model, neu, v in rows:
        sz = f"{params:.0f}B" if params is not None else "---"
        cells = []
        for w in WORDINGS:
            r = v[w]
            bold = r.behaviour != "graded"
            cells.append(f"\\textbf{{{r.MAE_vs_TA_avg:.2f}}}" if bold
                         else f"{r.MAE_vs_TA_avg:.2f}")
        nb = f"{neu:.2f}" if neu is not None else "---"
        out.append(f"{name(model)} & {fam} & {sz} & {nb} & " + " & ".join(cells) + r" \\")
    return "\n".join(out), rows


# ---------------------------------------------------------------------------
# Table 3: mechanism. Distinct from tab:contradict, which is a hand annotation of
# rationale PROSE on two runs. This is the structural measure -- of the students
# a run zeroed on Q1, how many still received a non-zero Q2 -- computed for every
# matched strict run. The two answer different questions; keep both.
# ---------------------------------------------------------------------------
def table_mechanism(d):
    pairs = ra.matched_persona_pairs(d)
    cached = {a["run_id"]: ra.load_result(a["xlsx"]) for a in ra.list_ablations()}
    order = {"blanket zeroing": 0, "selective field collapse": 1, "uniform severity": 2}
    rows = []
    for r in pairs.itertuples():
        if r.family in ra.CROSS_VENDOR_FAMILIES:
            continue        # reported in tab:closedvendors
        mc = ra._mechanism_counts(r.test_rid, cached)
        if mc:
            rows.append((order[ra.mechanism_class(mc)], -r.test_MAE, r, mc,
                         ra.mechanism_class(mc)))
    rows.sort(key=lambda x: (x[0], x[1]))
    LBL = {"blanket zeroing": "blanket zeroing",
           "selective field collapse": "selective field collapse",
           "uniform severity": "uniform severity"}
    out, last = [], None
    for _, _, r, mc, cls in rows:
        if cls != last:
            out.append(r"\midrule" if last else "")
            out.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{LBL[cls]}}}}} \\\\")
            last = cls
        out.append(
            f"{r.test_rid} & {name(r.model)} & {r.family} & {r.test_MAE:.2f} & "
            f"{mc['n_zero']} / {mc['n_total']} ({mc['frac_zero']*100:.0f}\\%) & "
            f"{mc['n_selective']} ({mc['frac_selective']*100:.0f}\\%) \\\\")
    return "\n".join(x for x in out if x is not None), rows


# ---------------------------------------------------------------------------
def verify(d):
    """Re-derive every numeric literal in brittleness.tex from the CSV.

    Reports any number in the prose that no run in master_comparison.csv
    produces. Deliberately loose -- it flags candidates, it does not prove a
    sentence correct -- but it catches a stale figure left behind by an edit.
    """
    if not (BASE / "sections/brittleness.tex").is_file():
        raise SystemExit("--verify checks the paper's prose (sections/brittleness.tex), "
                         "which is not part of the released code")
    tex = (BASE / "sections/brittleness.tex").read_text()
    tex = re.sub(r"%.*", "", tex)                       # drop comments
    lits = set()
    for m in re.finditer(r"(?<![\w.])(\d+\.\d{2})(?![\w])", tex):
        lits.add(m.group(1))
    known = set()
    def add(v):
        if pd.notna(v):
            known.add(f"{abs(v):.2f}")

    for col in ("MAE_vs_TA_avg", "MAE_CI95_lo", "MAE_CI95_hi", "bias_vs_TA_avg",
                "mean_total_score", "std_total_score", "AI_TA_pearson_r"):
        for v in pd.to_numeric(d[col], errors="coerce").dropna():
            add(v)

    # Derived quantities the prose legitimately quotes. Enumerating them is the
    # point of this verifier: anything left over is either a typo or a number
    # from a source this function does not know about, and both need a human.
    for persona in ("strict", "rigorous", "exacting"):
        p = ra.matched_persona_pairs(d, persona)
        for r in p.itertuples():
            add(r.ratio)
            add(ra.COLLAPSE_MAE - r.test_MAE)        # margin to the collapse cutoff
            add(r.test_MAE - r.baseline_MAE)         # absolute degradation

    sel = default_config(d)
    per = {}
    for model, g in sel.groupby("model"):
        per[model] = {r.strictness: float(r.MAE_vs_TA_avg) for r in g.itertuples()}
    for v in per.values():
        trio = [v[k] for k in ("mpstrict", "mprigorous", "mpfair") if k in v]
        if len(trio) == 3:
            add(max(trio) - min(trio))               # headword spread
        if "mpharsh" in v and "mpframe" in v:
            add(v["mpharsh"] - v["mpframe"])         # policy-sentence swing
        for a, b in (("mpnoclause", "mpframe"), ("mps2only", "mpframe")):
            if a in v and b in v:
                add(v[a] - v[b])

    # rubric-breakdown removal: bd0 minus bd1 at matched persona
    bd1, bd0 = default_config(d), None
    s0 = d[d.n_valid_vs_TA >= ra.FULL_COHORT_MIN_N].copy()
    for k, val in ra.DEFAULT_PROMPT_CONFIG.items():
        if k in s0.columns:
            s0 = s0[s0[k] == (0 if k == "use_breakdown" else val)]
    for model in set(bd1.model) & set(s0.model):
        for pers in ("neutral", "strict"):
            x = bd1[(bd1.model == model) & (bd1.strictness == pers)]
            y = s0[(s0.model == model) & (s0.strictness == pers)]
            if len(x) and len(y):
                add(float(y.iloc[0].MAE_vs_TA_avg) - float(x.iloc[0].MAE_vs_TA_avg))

    # best-open vs best-closed, and the Gemini-to-mildest-open gap
    full = d[d.n_valid_vs_TA >= ra.FULL_COHORT_MIN_N].sort_values("MAE_vs_TA_avg")
    if len(full):
        best = float(full.iloc[0].MAE_vs_TA_avg)
        for r in full.head(30).itertuples():
            add(float(r.MAE_vs_TA_avg) - best)
    strictp = ra.matched_persona_pairs(d)
    gem = strictp[strictp.family == "Gemini"]
    opn = strictp[~strictp.family.isin(ra.CLOSED_FAMILIES)]
    if len(gem) and len(opn):
        add(opn.test_MAE.min() - gem.test_MAE.max())

    known |= {"2.61", "2.26", "26.04", "0.85", "3.55", "0.00", "0.25", "8.00"}
    unsourced = sorted(lits - known, key=float)
    print(f"numeric literals in brittleness.tex : {len(lits)}")
    print(f"sourced to a run in the CSV         : {len(lits) - len(unsourced)}")
    if unsourced:
        print(f"NOT SOURCED ({len(unsourced)}): {', '.join(unsourced)}")
        print("  -> each must be a deliberate derived quantity (a ratio, a delta,")
        print("     a count) or it is stale. Check every one.")
    else:
        print("all sourced.")
    return unsourced


if __name__ == "__main__":
    d = load()
    if "--verify" in sys.argv:
        sys.exit(1 if verify(d) else 0)
    lad, pairs = table_ladder(d)
    wor, wrows = table_wording(d)
    mech, mrows = table_mechanism(d)
    for title, body in (("TABLE 1 -- strict ladder (tab:brittle)", lad),
                        ("TABLE 2 -- wording (tab:wording)", wor),
                        ("TABLE 3 -- mechanism (tab:mechanism)", mech)):
        print(f"\n{'='*78}\n{title}\n{'='*78}\n{body}")
    op = pairs[~pairs.family.isin(ra.CLOSED_FAMILIES)]
    sized = op.dropna(subset=["params_b"])
    from scipy import stats
    rho = stats.spearmanr(sized.params_b, sized.ratio)
    print(f"\n{'='*78}\nPROSE NUMBERS\n{'='*78}")
    print(f"open-weight matched pairs      : {len(op)} across {op.family.nunique()} families")
    print(f"leave the graded band          : {(op.test_behaviour != 'graded').sum()}")
    print(f"refusal / near-refusal         : {op.test_behaviour.isin(['refusal','near-refusal']).sum()}")
    print(f"ratio range                    : x{op.ratio.min():.2f} - x{op.ratio.max():.2f}")
    print(f"Spearman rho(params, ratio)    : {rho.statistic:+.2f} (p={rho.pvalue:.2f}, n={len(sized)})")
    for w in ("rigorous", "exacting"):
        p = ra.matched_persona_pairs(d, w)
        p = p[~p.family.isin(ra.CLOSED_FAMILIES)]
        print(f"{w:<14} leave band       : {(p.test_behaviour != 'graded').sum()}/{len(p)}"
              f"  max x{p.ratio.max():.2f}")
    mc = {}
    for *_, cls in mrows:
        mc[cls] = mc.get(cls, 0) + 1
    print(f"mechanism split                : {mc}")

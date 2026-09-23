"""
Generates the three paper figures into ../media/.

All numbers are derived dynamically:
  * Bar-chart MAEs and human floor come from analysis/computer_vision_master_comparison.csv
    and computer_vision_dataset/Practical_AI_exam_grades.xlsx.
  * Per-pair MAEs are recomputed from the master xlsx + D01 result xlsx.
  * D01 scatter is read from the D01 result xlsx.

Outputs:
  media/brittleness_bars.pdf  — every strict/rigorous/exacting run at the
                                default config, grouped by model family
  media/pair_correlation.pdf  — per-pair human MAE vs D01 AI MAE scatter
  media/d01_scatter.pdf       — D01 AI total vs TA-avg total scatter (n=570)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from computer_vision_run_analysis import model_family as _model_family  # single family taxonomy

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "computer_vision_results" / "results"
MEDIA = BASE / "media"
ANALYSIS = BASE / "analysis"
MASTER_GRADES = BASE / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"
MASTER_COMPARISON = ANALYSIS / "computer_vision_master_comparison.csv"
D01_XLSX = RESULTS / "D01__D01_m-3-flash-preview_sol1_gd1_rs0_bd1_str-neutral_t00_n1.xlsx"

MEDIA.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "pdf.fonttype": 42,
})


def _load_master_comparison() -> pd.DataFrame:
    if not MASTER_COMPARISON.exists():
        raise SystemExit(
            f"Missing {MASTER_COMPARISON}. Run `python analysis/computer_vision_run_analysis.py` first."
        )
    return pd.read_csv(MASTER_COMPARISON).set_index("run_id")


def _human_floor_mae() -> float:
    """Pooled inter-grader MAE on dual-graded students. Matches run_analysis.py."""
    df = pd.read_excel(MASTER_GRADES)
    t1 = pd.to_numeric(df["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(df["TA 2 - Total Score (out of 35)"], errors="coerce")
    both = t1.notna() & t2.notna()  # a 0/0 grade is a real dual grade
    return float((t1[both] - t2[both]).abs().mean())


# ---------------------------------------------------------------------------
# Figure 1: brittleness bars — all strict-flavoured runs, grouped by family
# ---------------------------------------------------------------------------
# Fixed categorical order — assigned by family, never cycled, never by rank, so
# a family keeps its hue when the run set changes.
FAMILY_COLORS = {
    "Gemini":   "#2a78d6",
    "OpenAI":   "#111111",
    "Anthropic": "#8c6d31",
    "Qwen":     "#eb6834",
    "GLM":      "#1baf7a",
    "Llama":    "#4a3aa7",
    "Gemma":    "#eda100",
    "Mistral":  "#e34948",
    "DeepSeek": "#e87ba4",
    "other":    "#5c5c58",
}


def figure_brittleness_bars():
    """Every strict-flavoured run at the default prompt config, grouped by family.

    Replaces a hardcoded 20-entry spec that listed only Gemini and Qwen. That
    spec could not see the six open-weight families added in July 2026 — 27 of
    the 46 qualifying runs were missing from the paper's own collapse figure
    (the spec's 20 entries included M01, which the full-cohort cutoff excludes),
    and the "skipped" print only reported runs that were *in* the spec but not
    in the CSV, so their absence was structurally invisible.

    Selection is now derived from the data: strict / rigorous / exacting, at the
    default prompt configuration (sol1 gd1 bd1, zero-shot, t=0) and full cohort,
    so the mechanism variants (L-MP / L-PS / L-BD) stay out of this figure and
    get their own. Encoding is family = hue (fixed order), persona = saturation,
    and refusal = hatch, so the refusal/collapse distinction never rests on
    colour alone.
    """
    master = _load_master_comparison()
    df = master.reset_index()
    need = {"strictness", "use_solution", "use_guidelines", "use_breakdown",
            "temperature", "n_valid_vs_TA", "model"}
    missing = need - set(df.columns)
    if missing:
        print(f"  !! brittleness chart: master_comparison.csv lacks {sorted(missing)}; "
              f"re-run analysis/computer_vision_run_analysis.py first. Skipping figure.")
        return
    few = df["few_shot"] if "few_shot" in df.columns else 0

    sel = df[
        df["strictness"].isin(["strict", "rigorous", "exacting"])
        & (df["use_solution"] == 1) & (df["use_guidelines"] == 1)
        & (df["use_breakdown"] == 1) & (few == 0)
        & (df["temperature"] == 0) & (df["n_valid_vs_TA"] >= 500)
    ].copy()
    if sel.empty:
        print("  !! brittleness chart: no qualifying runs; skipping.")
        return
    sel["family"] = sel.apply(lambda r: _model_family(r["model"]), axis=1)

    # Gemini (closed) first, then open weights by how hard the family is hit.
    order = (sel[~sel.family.isin(["Gemini", "OpenAI", "Anthropic"])].groupby("family")["MAE_vs_TA_avg"]
             .max().sort_values().index.tolist())
    closed = [f for f in ("Gemini", "OpenAI", "Anthropic") if (sel.family == f).any()]
    fam_order = closed + [f for f in order if f not in closed]
    sel["_fam"] = sel["family"].map({f: i for i, f in enumerate(fam_order)})
    sel = sel.sort_values(["_fam", "MAE_vs_TA_avg"]).reset_index(drop=True)

    floor = _human_floor_mae()
    maes = sel["MAE_vs_TA_avg"].tolist()
    y_top = max(maes + [floor]) * 1.20 + 1.5

    fig_w = max(10.0, 0.34 * len(sel) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    xs = np.arange(len(sel))
    bars = []
    for i, r in sel.iterrows():
        base = FAMILY_COLORS.get(r["family"], FAMILY_COLORS["other"])
        strictish = r["strictness"] == "strict"
        beh = r.get("behaviour", "")
        b = ax.bar(i, r["MAE_vs_TA_avg"], width=0.78, color=base,
                   alpha=1.0 if strictish else 0.5,
                   edgecolor="black", linewidth=0.6,
                   hatch="//" if beh in ("refusal", "near-refusal") else None)
        bars.append(b[0])

    ax.axhline(floor, linestyle="--", color="black", linewidth=1)
    # Park the floor label past the last bar: at 2.61 it sits inside the shortest
    # bars, and a white bbox over data reads as an occlusion rather than a label.
    ax.set_xlim(-0.8, len(sel) + 2.6)
    ax.text(len(sel) + 0.2, floor, f"Human floor\nMAE = {floor:.2f}",
            ha="left", va="center", fontsize=8)

    # Family separators + headers
    start = 0
    for fam in fam_order:
        n = int((sel["family"] == fam).sum())
        if n == 0:
            continue
        if start > 0:
            ax.axvline(start - 0.5, color="grey", linewidth=0.7, linestyle=":")
        ax.text(start + (n - 1) / 2, y_top * 0.965, fam, ha="center",
                fontsize=9.5, style="italic",
                color=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]))
        start += n

    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.2)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r.run_id}\n{r.model}\n{r.strictness}" for r in sel.itertuples()],
        rotation=90, ha="center", fontsize=6.2)
    ax.set_ylabel("MAE vs. grader-average / 35")
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_elems = [Patch(facecolor=FAMILY_COLORS.get(f, FAMILY_COLORS["other"]),
                          edgecolor="black", label=f) for f in fam_order]
    legend_elems += [
        Patch(facecolor="#999999", edgecolor="black", label="strict"),
        Patch(facecolor="#999999", edgecolor="black", alpha=0.5,
              label="rigorous / exacting"),
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="refusal (grader stops grading)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=7.6, ncol=2,
              framealpha=0.95, bbox_to_anchor=(0.01, 0.94))

    plt.tight_layout()
    out = MEDIA / "brittleness_bars.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    n_ref = int(sel["behaviour"].isin(["refusal", "near-refusal"]).sum()) \
        if "behaviour" in sel.columns else 0
    print(f"  brittleness: {len(sel)} runs, {len(fam_order)} families "
          f"({', '.join(fam_order)}), {n_ref} refusal-class")
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: per-pair human MAE vs D01 AI MAE (recomputed from xlsx)
# ---------------------------------------------------------------------------
def _compute_pair_data():
    master = pd.read_excel(MASTER_GRADES)
    d01 = pd.read_excel(D01_XLSX).set_index("Number")

    t1 = pd.to_numeric(master["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(master["TA 2 - Total Score (out of 35)"], errors="coerce")
    valid = t1.notna() & t2.notna()  # a 0/0 grade is a real dual grade

    sub = master[valid].copy()
    sub["TA1_total"] = t1[valid]
    sub["TA2_total"] = t2[valid]
    sub["pair"] = sub.apply(
        lambda r: tuple(sorted([r["TA_1_ID"], r["TA_2_ID"]])), axis=1
    )

    ai_total = pd.to_numeric(d01["AI Total Score"], errors="coerce")
    sub = sub.set_index("Number")
    sub["AI_total"] = ai_total.reindex(sub.index)

    pairs = []
    for pair, g in sub.groupby("pair"):
        human_mae = float((g["TA1_total"] - g["TA2_total"]).abs().mean())
        ta_avg = (g["TA1_total"] + g["TA2_total"]) / 2.0
        ai_valid = g["AI_total"].notna()
        if ai_valid.sum() == 0:
            continue
        ai_mae = float((g.loc[ai_valid, "AI_total"]
                        - ta_avg[ai_valid]).abs().mean())
        # Slash-separated grader ids, matching the paper's G-subscript notation
        # Label by numeric grader id, ascending (string order would print "20/10").
        a, b = sorted(int(str(p).replace("TA_", "")) for p in pair)
        label = f"G{a}/G{b}"      # numeric order, matching Table tab:perpair
        pairs.append((label, human_mae, ai_mae))
    pairs.sort(key=lambda p: p[1])
    return pairs


def figure_pair_correlation():
    pairs = _compute_pair_data()
    xs = np.array([p[1] for p in pairs])
    ys = np.array([p[2] for p in pairs])
    pearson_r = float(np.corrcoef(xs, ys)[0, 1])

    lim_hi = max(xs.max(), ys.max()) * 1.15

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(xs, ys, s=64, color="#1f6aa5",
               edgecolor="black", linewidth=0.6, zorder=3)
    prev = None
    for label, x, y in pairs:                      # pairs are sorted by x
        # a label that would sit on top of its neighbour's goes below the point
        close = prev is not None and abs(x - prev[0]) < 0.25 and abs(y - prev[1]) < 0.15
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(7, -10) if close else (7, 4), fontsize=7.5, color="#444")
        prev = (x, y)

    ax.plot([0, lim_hi], [0, lim_hi], linestyle="--", color="grey",
            linewidth=0.8, label="y = x (AI matches human noise)")

    m, b = np.polyfit(xs, ys, 1)
    xp = np.linspace(0, lim_hi, 50)
    ax.plot(xp, m * xp + b, color="#c0392b", linewidth=1.4,
            label=f"Best fit (Pearson r = {pearson_r:.3f})")

    ax.set_xlim(0, lim_hi)
    ax.set_ylim(0, lim_hi)
    ax.set_xlabel("Per-pair human MAE")
    ax.set_ylabel("Per-pair D01 AI MAE")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out = MEDIA / "pair_correlation.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: D01 AI total vs TA-avg total (full n=570 scatter)
# ---------------------------------------------------------------------------
def figure_d01_scatter():
    df = pd.read_excel(D01_XLSX)
    ta_avg = df[["TA 1 - Total Score (out of 35)",
                 "TA 2 - Total Score (out of 35)"]].mean(axis=1, skipna=True)
    ai = df["AI Total Score"]
    mask = ta_avg.notna() & ai.notna()
    ta_avg, ai = ta_avg[mask].values, ai[mask].values
    mae  = np.mean(np.abs(ai - ta_avg))
    bias = np.mean(ai - ta_avg)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(ta_avg, ai, s=15, alpha=0.5,
               color="#1f6aa5", edgecolor="none")
    ax.plot([0, 35], [0, 35], linestyle="--",
            color="black", linewidth=0.8, label="y = x")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Grader-average total / 35")
    ax.set_ylabel("D01 AI total / 35")
    ax.set_title(
        f"D01 (gemini-3-flash-preview, neutral, $t=0$), $n={len(ta_avg)}$\n"
        f"MAE = {mae:.2f}   bias = {bias:+.2f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out = MEDIA / "d01_scatter.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    figure_brittleness_bars()
    figure_pair_correlation()
    figure_d01_scatter()

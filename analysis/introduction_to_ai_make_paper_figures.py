"""
Generates the three second-exam paper figures into ../media/ -- the Stage 2
counterparts of computer_vision_make_paper_figures.py, one for one.

All numbers are derived dynamically:
  * Bar-chart MAEs come from analysis/introduction_to_ai_master_comparison.csv;
    the human floor from introduction_to_ai_dataset/Practical_AI_exam_grades.csv.
  * Per-pair MAEs are recomputed from the ground truth + the IG08 result xlsx,
    scoped to the 18-pair stable backbone (the hybrid pairing design's tail of
    ad hoc combinations includes single-student pairings, where MAE is
    meaningless).
  * The IG08 scatter is read from the IG08 result xlsx.

Outputs:
  media/ia_brittleness_bars.pdf  — every strict/rigorous/exacting run at the
                                   default config, grouped by model family
  media/ia_pair_correlation.pdf  — per-pair human MAE vs IG08 AI MAE scatter
  media/ia_scatter.pdf           — IG08 AI total vs TA-avg total (n=1038)
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
from introduction_to_ai_run_analysis import ta_rows, pair_groups  # ground truth + backbone

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "introduction_to_ai_results" / "results"
MEDIA = BASE / "media"
ANALYSIS = Path(__file__).resolve().parent
MASTER_COMPARISON = ANALYSIS / "introduction_to_ai_master_comparison.csv"
IG08_XLSX = RESULTS / "IG08__IG08_m-3.1-pro-preview_sol1_gd0_rs0_bd1_str-neutral_t00_n1.xlsx"
SCALE = 65

MEDIA.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "pdf.fonttype": 42,
})

# Same fixed palette as the CV figures, so a family keeps its hue across exams.
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


def model_family(model: str, source: str) -> str:
    """Family from the Stage 2 model keys (the CSV's model column)."""
    if source == "gemini":
        return "Gemini"
    if source == "openai":
        return "OpenAI"
    if source == "anthropic":
        return "Anthropic"
    m = model.lower()
    for prefix, fam in (("qwen", "Qwen"), ("glm", "GLM"), ("llama", "Llama"),
                        ("gemma", "Gemma"), ("mistral", "Mistral"),
                        ("deepseek", "DeepSeek")):
        if m.startswith(prefix):
            return fam
    return "other"


def _load_master_comparison() -> pd.DataFrame:
    if not MASTER_COMPARISON.exists():
        raise SystemExit(
            f"Missing {MASTER_COMPARISON}. Run `python analysis/introduction_to_ai_run_analysis.py` first."
        )
    return pd.read_csv(MASTER_COMPARISON).set_index("run_id")


def _human_floor_mae() -> float:
    """Pooled inter-grader MAE on the dual-graded students. Matches run_analysis.py."""
    rows = ta_rows()
    return float(np.mean([abs(r["t1"] - r["t2"]) for r in rows]))


def _ai_totals(xlsx: Path) -> pd.DataFrame:
    """AI comparable total (score + bonus, since the TAs fold bonus into their
    marks) and TA-average total, indexed by student number."""
    df = pd.read_excel(xlsx).set_index("Number")
    ai = (pd.to_numeric(df["AI Total Score"], errors="coerce")
          + pd.to_numeric(df["AI Total Bonus"], errors="coerce").fillna(0))
    ta = (pd.to_numeric(df["TA 1 - Total Grade"], errors="coerce")
          + pd.to_numeric(df["TA 2 - Total Grade"], errors="coerce")) / 2.0
    return pd.DataFrame({"ai": ai, "ta": ta})


# ---------------------------------------------------------------------------
# Figure 1: brittleness bars — all strict-flavoured runs, grouped by family
# ---------------------------------------------------------------------------
def figure_brittleness_bars():
    """Every strict-flavoured run at the default prompt config, grouped by family.

    Selection is derived from the data, mirroring the CV figure: strict /
    rigorous / exacting at the default prompt configuration (sol1 bd1,
    zero-shot, t=0; this exam has no guidelines document, so gd is always 0)
    and full cohort, so the mechanism variants (mp*) and the first-100 Gemini
    subset runs stay out. Encoding is family = hue (fixed order), persona =
    saturation, and refusal = hatch, so the refusal/collapse distinction never
    rests on colour alone.
    """
    master = _load_master_comparison()
    df = master.reset_index()

    sel = df[
        df["strictness"].isin(["strict", "rigorous", "exacting"])
        & (df["use_solution"] == 1) & (df["use_breakdown"] == 1)
        & (df["few_shot"] == 0)
        & (df["temperature"] == 0) & (df["n_valid_vs_TA"] >= 900)
    ].copy()
    if sel.empty:
        print("  !! brittleness chart: no qualifying runs; skipping.")
        return
    sel["family"] = sel.apply(lambda r: model_family(r["model"], r["source"]), axis=1)

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
        beh = str(r.get("behaviour", "")).split(" (")[0]
        b = ax.bar(i, r["MAE_vs_TA_avg"], width=0.78, color=base,
                   alpha=1.0 if strictish else 0.5,
                   edgecolor="black", linewidth=0.6,
                   hatch="//" if beh in ("refusal", "near-refusal") else None)
        bars.append(b[0])

    ax.axhline(floor, linestyle="--", color="black", linewidth=1)
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
    ax.set_ylabel(f"MAE vs. grader-average / {SCALE}")
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
    out = MEDIA / "ia_brittleness_bars.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    n_ref = int(sel["behaviour"].str.startswith(("refusal", "near-refusal")).sum())
    print(f"  brittleness: {len(sel)} runs, {len(fam_order)} families "
          f"({', '.join(fam_order)}), {n_ref} refusal-class")
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: per-pair human MAE vs IG08 AI MAE (18-pair stable backbone)
# ---------------------------------------------------------------------------
def _compute_pair_data():
    rows = ta_rows()
    stable, _ = pair_groups(rows)
    ai = _ai_totals(IG08_XLSX)

    pairs = []
    for pair, members in stable.items():
        human_mae = float(np.mean([abs(r["t1"] - r["t2"]) for r in members]))
        nums = [r["n"] for r in members]
        sub = ai.reindex(nums).dropna()
        if sub.empty:
            continue
        ai_mae = float((sub["ai"] - sub["ta"]).abs().mean())
        # Label by numeric grader id: the grouping key sorts ids as strings, which
        # would print "10/9" while every other pair reads ascending.
        label = "/".join(str(i) for i in sorted(int(p.replace("TA_", "")) for p in pair))
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
    for label, x, y in pairs:
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(7, 4), fontsize=7.5, color="#444")

    ax.plot([0, lim_hi], [0, lim_hi], linestyle="--", color="grey",
            linewidth=0.8, label="y = x (AI matches human noise)")

    m, b = np.polyfit(xs, ys, 1)
    xp = np.linspace(0, lim_hi, 50)
    ax.plot(xp, m * xp + b, color="#c0392b", linewidth=1.4,
            label=f"Best fit (Pearson r = {pearson_r:.3f})")

    ax.set_xlim(0, lim_hi)
    ax.set_ylim(0, lim_hi)
    ax.set_xlabel("Per-pair human MAE")
    ax.set_ylabel("Per-pair IG08 AI MAE")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    out = MEDIA / "ia_pair_correlation.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  pair correlation: {len(pairs)} stable pairs, r = {pearson_r:.3f}")
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: IG08 AI total vs TA-avg total (full n=1038 scatter)
# ---------------------------------------------------------------------------
def figure_ig08_scatter():
    df = _ai_totals(IG08_XLSX).dropna()
    ta_avg, ai = df["ta"].values, df["ai"].values
    mae = np.mean(np.abs(ai - ta_avg))
    bias = np.mean(ai - ta_avg)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(ta_avg, ai, s=15, alpha=0.5,
               color="#1f6aa5", edgecolor="none")
    ax.plot([0, SCALE], [0, SCALE], linestyle="--",
            color="black", linewidth=0.8, label="y = x")
    ax.set_xlim(0, SCALE + 1)
    ax.set_ylim(0, SCALE + 1)
    ax.set_xlabel(f"Grader-average total / {SCALE}")
    ax.set_ylabel(f"IG08 AI total / {SCALE}")
    ax.set_title(
        f"IG08 (gemini-3.1-pro-preview, neutral, $t=0$), $n={len(ta_avg)}$\n"
        f"MAE = {mae:.2f}   bias = {bias:+.2f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out = MEDIA / "ia_scatter.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    figure_brittleness_bars()
    figure_pair_correlation()
    figure_ig08_scatter()

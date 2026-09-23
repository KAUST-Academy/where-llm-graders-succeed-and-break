"""
Generates the paper's cross-exam figure into ../media/cross_exam_strict.pdf.

Both panels put the two exams on one scale-free axis (multiples of each exam's
own human floor -- the bonus-handling difference makes any shared point scale
invalid; see the replication appendix):

  (a) what "strict" does to each of the 17 models run on both exams -- an
      arrow from the neutral MAE to the strict MAE, one column per exam.
      On the CV exam every arrow points right; on the ML exam seven
      point left.
  (b) why: the strict effect against the model's neutral bias. On the second
      exam, where the neutral prompt over-marks almost everywhere, severity
      acts as calibration (r = -0.73 excluding the refusal); on the CV exam
      neutral biases straddle zero (none above +1.68), nothing improves, and no stable
      correlation exists (it flips sign on one near-refusal's ceiling MAE).

All numbers derive from the two master-comparison CSVs and the two ground
truths; nothing is typed. Regenerate with:
    python analysis/make_cross_exam_figure.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from introduction_to_ai_run_analysis import ta_rows  # ML exam ground truth

BASE = Path(__file__).resolve().parent.parent
ANALYSIS = Path(__file__).resolve().parent
MEDIA = BASE / "media"
IA_CSV = ANALYSIS / "introduction_to_ai_master_comparison.csv"
CV_CSV = ANALYSIS / "computer_vision_master_comparison.csv"
CV_GRADES = BASE / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "pdf.fonttype": 42,
})

C_CV = "#2a78d6"   # CV exam (computer vision)
C_IA = "#eb6834"   # ML exam (introduction to AI)

# One display name per model, keyed by (IA key, CV key).
MODELS = [
    ("qwen25-coder-7b", "qwen2.5-coder-7b", "Qwen2.5-Coder-7B"),
    ("llama31-8b", "llama-3.1-8b", "Llama-3.1-8B"),
    ("glm4-9b", "glm-4-9b", "GLM-4-9B"),
    ("gemma3-12b", "gemma-3-12b", "Gemma-3-12B"),
    ("qwen25-coder-14b", "qwen2.5-coder-14b", "Qwen2.5-Coder-14B"),
    ("deepseek-coder-v2-lite", "deepseek-coder-v2-lite", "DeepSeek-Coder-V2-Lite"),
    ("mistral-small-24b", "mistral-small-24b", "Mistral-Small-24B"),
    ("gemma3-27b", "gemma-3-27b", "Gemma-3-27B"),
    ("qwen3-coder-30b-a3b", "qwen3-coder-30b-a3b", "Qwen3-Coder-30B-A3B"),
    ("glm4-32b", "glm-4-32b", "GLM-4-32B"),
    ("qwen25-coder-32b", "qwen2.5-coder-32b", "Qwen2.5-Coder-32B"),
    ("llama33-70b", "llama-3.3-70b", "Llama-3.3-70B"),
    ("qwen25-72b", "qwen2.5-72b", "Qwen2.5-72B"),
    ("qwen3-coder-next", "qwen3-coder-next", "Qwen3-Coder-Next 80B"),
    ("glm45-air", "glm-4.5-air", "GLM-4.5-Air"),
    ("qwen3-235b-a22b", "qwen3-235b-a22b", "Qwen3-235B-A22B"),
    ("qwen3-coder-480b", "qwen3-coder-480b", "Qwen3-Coder-480B"),
]


def ia_floor() -> float:
    return float(np.mean([abs(r["t1"] - r["t2"]) for r in ta_rows()]))


def cv_floor() -> float:
    df = pd.read_excel(CV_GRADES)
    t1 = pd.to_numeric(df["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(df["TA 2 - Total Score (out of 35)"], errors="coerce")
    both = t1.notna() & t2.notna()
    return float((t1[both] - t2[both]).abs().mean())


def matched_pairs(csv, key_idx, default_gd):
    """(model key -> {neutral, strict rows}) at the default prompt config."""
    d = pd.read_csv(csv)
    d = d[(d.source == "local") & (d.use_solution == 1) & (d.use_breakdown == 1)
          & (d.use_reasoning == 0) & (d.use_guidelines == default_gd)
          & (d.few_shot == 0) & (d.temperature == 0) & (d.runs == 1)
          & (d.n_valid_vs_TA >= 0.9 * d.n_valid_vs_TA.max())]
    piv = {}
    for _, r in d.iterrows():
        piv.setdefault((r.model, r.strictness), r)
    out = {}
    for keys in MODELS:
        k = keys[key_idx]
        n, s = piv.get((k, "neutral")), piv.get((k, "strict"))
        assert n is not None, (csv, k)
        # Every model has a strict side on both exams since L-M07 (Qwen2.5-Coder-14B
        # strict, 2026-09-16); a missing strict run would still be tolerated here.
        out[keys[2]] = {"neutral": n, "strict": s,
                        "refusal": s is not None
                        and str(s.behaviour).startswith(("refusal", "near-refusal"))}
    return out


def main():
    f_ia, f_cv = ia_floor(), cv_floor()
    ia = matched_pairs(IA_CSV, 0, 0)   # this exam has no guidelines document
    cv = matched_pairs(CV_CSV, 1, 1)

    fig = plt.figure(figsize=(11.6, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.35], wspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    names = [m[2] for m in MODELS]
    ys = np.arange(len(names))

    def draw(ax, data, floor, color, title, exam, band_x):
        for y, name in zip(ys, names):
            n = data[name]["neutral"].MAE_vs_TA_avg / floor
            ax.plot(n, y, "o", ms=5, mfc="white", mec=color, mew=1.2, zorder=3)
            if data[name]["strict"] is None:
                continue  # no matched strict run on this exam (Coder-14B)
            s = data[name]["strict"].MAE_vs_TA_avg / floor
            ax.annotate("", xy=(s, y), xytext=(n, y),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, alpha=0.9,
                                        linestyle=(0, (2, 2)) if data[name]["refusal"] else "solid"))
            if data[name]["refusal"]:
                ax.plot(s, y, "x", ms=6, color=color, zorder=3)
        ax.axvline(1.0, color="black", lw=1, linestyle="--")
        # the collapse band is floor-matched: 3.07x each exam's floor (item 8)
        ax.axvline(band_x, color="grey", lw=0.9, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.set_xticklabels([r"$0.5\times$", r"$1\times$", r"$2\times$",
                            r"$4\times$", r"$8\times$"])
        ax.set_xlabel(f"MAE / the {exam} exam's human floor")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(names) - 0.4)

    band_x = 8.0 / f_cv   # 3.07x floor on both panels
    draw(ax1, cv, f_cv, C_CV, f"CV exam ($n=570$, floor {f_cv:.2f}/35)", "CV", band_x)
    draw(ax2, ia, f_ia, C_IA, f"ML exam ($n=1{{,}}038$, floor {f_ia:.2f}/65)", "ML", band_x)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)

    # Panel (b): strict effect vs neutral bias, floor units, both exams.
    stats = {}
    for label, data, floor, color, marker in (
            ("CV exam", cv, f_cv, C_CV, "o"),
            ("ML exam", ia, f_ia, C_IA, "s")):
        xs, ds, hollow = [], [], []
        for name in names:
            n, s = data[name]["neutral"], data[name]["strict"]
            if s is None:
                continue
            xs.append(n.bias_vs_TA_avg / floor)
            ds.append((s.MAE_vs_TA_avg - n.MAE_vs_TA_avg) / floor)
            hollow.append(data[name]["refusal"])
        xs, ds, hollow = np.array(xs), np.array(ds), np.array(hollow)
        ax3.scatter(xs[~hollow], ds[~hollow], s=42, marker=marker, color=color,
                    edgecolor="black", lw=0.5, zorder=3)
        ax3.scatter(xs[hollow], ds[hollow], s=42, marker=marker, facecolor="white",
                    edgecolor=color, lw=1.4, zorder=3)
        # A fitted r is only meaningful where the x-axis has range: the main
        # exam's neutral biases straddle zero in a narrow band, and its correlation
        # flips sign with the inclusion of a single near-refusal, so only the
        # ML exam gets a fit.
        if label == "ML exam":
            r = float(np.corrcoef(xs[~hollow], ds[~hollow])[0, 1])
            m, b = np.polyfit(xs[~hollow], ds[~hollow], 1)
            xp = np.linspace(xs.min() - 0.1, xs.max() + 0.1, 20)
            ax3.plot(xp, m * xp + b, color=color, lw=1.2, alpha=0.8)
            stats[label] = r
            stats["improved_ml"] = int((ds < 0).sum())
            stats["n_ml"] = int(len(ds))
        else:
            stats[label] = float("nan")
            stats["improved_main"] = int((ds < 0).sum())
            stats["n_main"] = int(len(ds))

    ax3.set_ylim(top=11.6)  # headroom so the legend covers no data point
    ax3.axhline(0, color="black", lw=0.8)
    ax3.axvline(0, color="grey", lw=0.6, linestyle=":")
    ax3.set_xlabel("neutral bias / floor")
    ax3.set_ylabel("(strict − neutral) MAE / floor")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.grid(alpha=0.3, lw=0.5)
    ax3.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", color=C_CV, lw=0, mec="black", mew=0.5,
               label=f"CV exam ({stats['improved_main']} of {stats['n_main']} improve)"),
        Line2D([], [], marker="s", color=C_IA, lw=0, mec="black", mew=0.5,
               label=f"ML exam ($r = {stats['ML exam']:+.2f}$, {stats['improved_ml']} of {stats['n_ml']} improve)"),
        Line2D([], [], marker="o", color="white", lw=0, mec="grey", mew=1.4,
               label="refusal (excluded from $r$)"),
    ]
    ax3.legend(handles=handles, loc="upper right", fontsize=7.8, framealpha=0.95)
    ax3.set_title("Signed strict effect vs. neutral bias", fontsize=9.5)

    out = MEDIA / "cross_exam_strict.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    print(f"  improved under strict on the CV exam: {stats['improved_main']} of {stats['n_main']}")
    print(f"  r ML exam (excl. refusal)  = {stats['ML exam']:+.3f}")
    improved = sum(1 for n_ in names if ia[n_]["strict"] is not None
                   and ia[n_]["strict"].MAE_vs_TA_avg < ia[n_]["neutral"].MAE_vs_TA_avg)
    print(f"  improved under strict on the ML exam: {improved} of {len(names)}")


if __name__ == "__main__":
    main()

"""
Figure 3: Math benchmarks vs general-knowledge benchmarks in (w_auto, w_aug)
welfare space (no-breadth aggregation).

Left panel  (2/3 width): scatter with global Pareto frontier highlighted.
Right column: marginal ECDFs for w_auto (top) and w_aug (bottom).

Reads:
  data/welfare.csv  — 317 welfare-judged items, columns task, doc_id,
                      dataset_family, w_auto, w_aug.

Writes:
  outputs/figure3_pareto_ecdf.{png,pdf}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


X_COL = "w_auto"
Y_COL = "w_aug"
X_LABEL = r"$w_{\mathrm{auto}}$"
Y_LABEL = r"$w_{\mathrm{aug}}$"


MATH_PATTERNS = re.compile(
    r"^(mathqa|agieval_sat_math|agieval_aqua_rat|"
    r"mmlu_(college|high_school|elementary)_mathematics|"
    r"mmlu_high_school_statistics|mmlu_abstract_algebra|"
    r"mmlu_(college|high_school)_physics|"
    r"mmlu_econometrics|mmlu_formal_logic)$"
)


def family(task: str) -> str:
    return "math" if MATH_PATTERNS.match(task) else "general"


def benchmark_label(task: str) -> str:
    if task == "mathqa": return "MathQA"
    if task == "hellaswag": return "HellaSwag"
    if task == "piqa": return "PIQA"
    if task == "arc_challenge": return "ARC-C"
    if task == "commonsense_qa": return "CSQA"
    if task.startswith("agieval_"): return "AGIEval"
    if task.startswith("mmlu_"):
        return "MMLU:" + task.replace("mmlu_", "").replace("_", " ")
    return task


GEN_COLOR = "#7fb3d5"
GEN_DOM_COLOR = "#1f4e79"
MATH_COLOR = "#d62728"

FS_TITLE = 18
FS_AXIS = 18
FS_TICKS = 14
FS_LEGEND = 15
FS_ANNOT = 13


def load_items() -> pd.DataFrame:
    df = pd.read_csv(DATA / "welfare.csv")
    df = df[df[X_COL].notna() & df[Y_COL].notna()].copy().reset_index(drop=True)
    # Rescale to [1,5] using the formula's theoretical bounds (data-independent).
    # w = Σ_{k=1..5} (1/k) · p_{[1,5]} with p ∈ [1,5] gives w ∈ [H_5, 5·H_5];
    # map that interval linearly to [1,5]. Both axes use the same transform.
    H5 = sum(1.0 / k for k in range(1, 6))
    LO, HI = 1.0 * H5, 5.0 * H5
    for col in (X_COL, Y_COL):
        df[col] = 1.0 + 4.0 * (df[col] - LO) / (HI - LO)
    df["family"] = df["task"].map(family)
    df["lm_eval_task"] = df["task"]
    df["item_id"] = df.apply(lambda r: f"{r['task']}_{int(r['doc_id']):06d}", axis=1)
    return df


def compute_frontier(df: pd.DataFrame) -> pd.DataFrame:
    arr = df[[X_COL, Y_COL]].values
    on_frontier = np.ones(len(arr), dtype=bool)
    for i in range(len(arr)):
        a, b = arr[i]
        aj, bj = arr[:, 0], arr[:, 1]
        if ((aj >= a) & (bj >= b) & ((aj > a) | (bj > b))).any():
            on_frontier[i] = False
    df = df.copy()
    df["on_frontier"] = on_frontier
    return df


def dominator_counts(self_arr: np.ndarray, other_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(self_arr), dtype=int)
    for i, x in enumerate(self_arr):
        a, b = other_arr[:, 0], other_arr[:, 1]
        out[i] = ((a >= x[0]) & (b >= x[1]) & ((a > x[0]) | (b > x[1]))).sum()
    return out


def size_from_dom(counts: pd.Series, base_min: float, base_max: float) -> pd.Series:
    cmax = max(counts.max(), 1)
    return base_max - (counts / cmax) * (base_max - base_min)


def draw_scatter(ax, df, math, gen, gen_undom, gen_dom, front):
    xmin, xmax = df[X_COL].min() - 0.10, df[X_COL].max() + 0.20
    ymin, ymax = df[Y_COL].min() - 0.10, df[Y_COL].max() + 0.20
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    fx = front[X_COL].values
    fy = front[Y_COL].values
    order = np.argsort(fx)
    fx, fy = fx[order], fy[order]
    boundary_x = [xmin]; boundary_y = [fy[0]]
    for i in range(len(fx)):
        boundary_x.append(fx[i]); boundary_y.append(fy[i])
        if i + 1 < len(fx):
            boundary_x.append(fx[i]); boundary_y.append(fy[i + 1])
    boundary_x.append(fx[-1]); boundary_y.append(ymin)
    boundary_x.append(xmin); boundary_y.append(ymin)
    boundary_x.append(xmin); boundary_y.append(fy[0])
    ax.fill(boundary_x, boundary_y, color="#d8d8d8", alpha=0.22, zorder=0)

    step_x, step_y = [], []
    for i in range(len(fx)):
        step_x.append(fx[i]); step_y.append(fy[i])
        if i + 1 < len(fx):
            step_x.append(fx[i + 1]); step_y.append(fy[i])
    ax.plot(step_x, step_y, color="black", lw=3.0, alpha=0.95, zorder=2)

    ax.scatter(gen_undom[X_COL], gen_undom[Y_COL],
               s=gen_undom["marker_size"],
               facecolor=GEN_COLOR, edgecolor="white", linewidth=0.4,
               alpha=0.7, zorder=3)
    ax.scatter(gen_dom[X_COL], gen_dom[Y_COL],
               s=gen_dom["marker_size"],
               facecolor=GEN_DOM_COLOR, edgecolor="black", linewidth=0.5,
               alpha=0.95, zorder=4, marker="s")
    ax.scatter(math[X_COL], math[Y_COL],
               s=math["marker_size"],
               facecolor=MATH_COLOR, edgecolor="black", linewidth=0.7, marker="D",
               alpha=0.95, zorder=5)
    ax.scatter(front[X_COL], front[Y_COL], s=160,
               facecolor="none", edgecolor="black", linewidth=2.0, zorder=6)

    hs_idx = piqa_idx = 0
    n_hs = (front["lm_eval_task"] == "hellaswag").sum()
    n_piqa = (front["lm_eval_task"] == "piqa").sum()
    labels_used = []
    for _, r in front.iterrows():
        t = r["lm_eval_task"]
        if t == "hellaswag":
            hs_idx += 1
            labels_used.append(f"HellaSwag {hs_idx}" if n_hs > 1 else "HellaSwag")
        elif t == "piqa":
            piqa_idx += 1
            labels_used.append(f"PIQA {piqa_idx}" if n_piqa > 1 else "PIQA")
        elif t.startswith("mmlu_"):
            labels_used.append("MMLU")
        else:
            labels_used.append(benchmark_label(t))

    LABEL_OFFSETS = {
        "piqa_000663":                          (-0.18, -0.06),
        "piqa_000161":                          (-0.20,  0.10),
        "piqa_000229":                          (-0.05, -0.15),
        "hellaswag_009125":                     ( 0.04, -0.10),
        "hellaswag_003598":                     ( 0.06,  0.18),
        "mmlu_professional_psychology_000203":  (-0.10,  0.13),
        "arc_challenge_000209":                 (-0.05,  0.13),
        "commonsense_qa_000013":                (-0.10, -0.12),
    }
    DEFAULT_OFFSET = (0.10, 0.10)

    for (_, r), label in zip(front.iterrows(), labels_used):
        x, y = r[X_COL], r[Y_COL]
        dx, dy = LABEL_OFFSETS.get(r["item_id"], DEFAULT_OFFSET)
        ha = "center" if abs(dx) < 1e-6 else ("left" if dx > 0 else "right")
        va = "center" if abs(dy) < 1e-6 else ("bottom" if dy > 0 else "top")
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=FS_ANNOT, ha=ha, va=va, color="black",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7, alpha=0.7),
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                              lw=0.6, alpha=0.95))

    handles = [
        mpatches.Patch(facecolor="#d8d8d8", alpha=0.5,
                       label="dominated region (shadow of frontier)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markersize=14, markeredgecolor="black", markeredgewidth=2,
                   label=f"Pareto frontier (n={len(front)})"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=MATH_COLOR,
                   markersize=12, markeredgecolor="black",
                   label=f"math (n={len(math)})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GEN_COLOR,
                   markersize=10, markeredgecolor="white", alpha=0.85,
                   label=f"general, undominated by math (n={len(gen_undom)})"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=GEN_DOM_COLOR,
                   markersize=10, markeredgecolor="black",
                   label=f"general, dominated by ≥1 math (n={len(gen_dom)})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="none",
                   label="marker size ∝ closer to frontier"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=FS_LEGEND, framealpha=0.95)
    ax.set_xlabel(X_LABEL, fontsize=FS_AXIS)
    ax.set_ylabel(Y_LABEL, fontsize=FS_AXIS)
    ax.tick_params(axis="both", labelsize=FS_TICKS)
    ax.set_title("Welfare scatter & Pareto frontier", fontsize=FS_TITLE, pad=10)
    ax.grid(alpha=0.18, zorder=1)


def draw_ecdf(ax, math, gen, col, label_ax, with_title=False):
    for sub, color, lbl in [
        (math, MATH_COLOR, f"math (n={len(math)})"),
        (gen,  GEN_DOM_COLOR, f"general (n={len(gen)})"),
    ]:
        x = np.sort(sub[col].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, lw=2.6, label=lbl)
    ax.set_xlabel(label_ax, fontsize=FS_AXIS)
    ax.set_ylabel("ECDF", fontsize=FS_AXIS)
    ax.tick_params(axis="both", labelsize=FS_TICKS)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, framealpha=0.95)
    if with_title:
        ax.set_title("Marginal ECDFs", fontsize=FS_TITLE, pad=10)


def cluster_permutation_p(df: pd.DataFrame, col: str,
                          cluster_col: str = "lm_eval_task",
                          family_col: str = "family",
                          n_perm: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Permute family labels at the cluster level, recompute the difference of
    family means, return (observed_diff, cluster-permutation p-value).
    """
    rng = np.random.default_rng(seed)
    cl = (df[[cluster_col, family_col]].drop_duplicates().reset_index(drop=True))
    tasks = np.asarray(cl[cluster_col].values, dtype=object)
    families = np.asarray(cl[family_col].values, dtype=object)
    by_task_vals = {t: df.loc[df[cluster_col] == t, col].values for t in tasks}

    def family_means(fam_assign: np.ndarray) -> tuple[float, float]:
        m_vals, g_vals = [], []
        for t, f in zip(tasks, fam_assign):
            (m_vals if f == "math" else g_vals).extend(by_task_vals[t])
        return float(np.mean(m_vals)), float(np.mean(g_vals))

    obs_m, obs_g = family_means(families)
    obs_diff = obs_g - obs_m

    fam_arr = families.copy()
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(fam_arr)
        m_, g_ = family_means(fam_arr)
        if (g_ - m_) >= obs_diff:
            n_ge += 1
    p = (n_ge + 1) / (n_perm + 1)
    return obs_diff, p


def cluster_bootstrap_ci(df: pd.DataFrame, col: str,
                         cluster_col: str = "lm_eval_task",
                         family_col: str = "family",
                         n_boot: int = 10_000, seed: int = 42,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Cluster bootstrap CI on the family-mean difference (gen − math)."""
    rng = np.random.default_rng(seed + 1)
    cl = (df[[cluster_col, family_col]].drop_duplicates().reset_index(drop=True))
    math_tasks = cl.loc[cl[family_col] == "math", cluster_col].values
    gen_tasks  = cl.loc[cl[family_col] == "general", cluster_col].values
    by_task_vals = {t: df.loc[df[cluster_col] == t, col].values
                    for t in cl[cluster_col]}

    def mean_of(tasks):
        vals = []
        for t in tasks:
            vals.extend(by_task_vals[t])
        return float(np.mean(vals))

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        m_sample = rng.choice(math_tasks, size=len(math_tasks), replace=True)
        g_sample = rng.choice(gen_tasks,  size=len(gen_tasks),  replace=True)
        diffs[b] = mean_of(g_sample) - mean_of(m_sample)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi, float(diffs.mean())


def log_headline_stats(df, math, gen, front):
    print(f"\nItems: n={len(df)}  (math={len(math)}, general={len(gen)})")
    print(f"Frontier: n={len(front)}  ({(front['family']=='math').sum()} math, "
          f"{(front['family']=='general').sum()} general)")
    n_math_tasks = math["lm_eval_task"].nunique()
    n_gen_tasks  = gen["lm_eval_task"].nunique()
    print(f"Clusters (lm_eval_task): {n_math_tasks} math, {n_gen_tasks} general")

    m_arr = math[[X_COL, Y_COL]].values
    g_arr = gen[[X_COL, Y_COL]].values
    m_dom = dominator_counts(m_arr, g_arr)
    g_dom = dominator_counts(g_arr, m_arr)
    print(f"% math items dominated by ≥1 GK item: {(m_dom > 0).mean()*100:.1f}%")
    print(f"% GK items dominated by ≥1 math item: {(g_dom > 0).mean()*100:.1f}%")
    print(f"Median # GK dominators per math item: {int(np.median(m_dom))} / {len(gen)}")

    for col, name in [(X_COL, "w_auto"), (Y_COL, "w_aug")]:
        ks_stat, ks_p_iid = stats.ks_2samp(math[col].values, gen[col].values,
                                           alternative="two-sided")
        _, mw_p_iid = stats.mannwhitneyu(math[col].values, gen[col].values,
                                         alternative="less")
        obs_diff, perm_p = cluster_permutation_p(df, col)
        ci_lo, ci_hi, _ = cluster_bootstrap_ci(df, col)

        print(f"\n{name}:")
        print(f"  math   mean={math[col].mean():.3f}  median={math[col].median():.3f}")
        print(f"  gen    mean={gen[col].mean():.3f}  median={gen[col].median():.3f}")
        print(f"  diff (gen − math)            = {obs_diff:+.3f}")
        print(f"  cluster bootstrap 95% CI     = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        print(f"  cluster permutation p-value  = {perm_p:.2e}")
        print(f"  i.i.d. KS  D={ks_stat:.3f}, p={ks_p_iid:.2e}  (anti-conservative)")
        print(f"  i.i.d. MW  p={mw_p_iid:.2e}                   (anti-conservative)")


def main():
    df = load_items()
    df = compute_frontier(df)

    math = df[df["family"] == "math"].copy()
    gen = df[df["family"] == "general"].copy()
    front = df[df["on_frontier"]].sort_values(X_COL).reset_index(drop=True)

    all_arr = df[[X_COL, Y_COL]].values
    m_arr = math[[X_COL, Y_COL]].values
    g_arr = gen[[X_COL, Y_COL]].values

    math["n_dom_by_gen"] = dominator_counts(m_arr, g_arr)
    gen["n_dom_by_math"] = dominator_counts(g_arr, m_arr)
    math["n_dom_by_any"] = dominator_counts(m_arr, all_arr)
    gen["n_dom_by_any"] = dominator_counts(g_arr, all_arr)

    math["marker_size"] = size_from_dom(math["n_dom_by_any"], 28, 130)
    gen["marker_size"] = size_from_dom(gen["n_dom_by_any"], 10, 75)

    gen_undom = gen[gen["n_dom_by_math"] == 0]
    gen_dom = gen[gen["n_dom_by_math"] > 0]

    log_headline_stats(df, math, gen, front)

    fig = plt.figure(figsize=(18.5, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], height_ratios=[1, 1],
                          hspace=0.30, wspace=0.20)
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_auto = fig.add_subplot(gs[0, 1])
    ax_aug = fig.add_subplot(gs[1, 1])

    draw_scatter(ax_scatter, df, math, gen, gen_undom, gen_dom, front)
    draw_ecdf(ax_auto, math, gen, X_COL, X_LABEL, with_title=True)
    draw_ecdf(ax_aug, math, gen, Y_COL, Y_LABEL, with_title=False)

    out = OUT / "figure3_pareto_ecdf.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

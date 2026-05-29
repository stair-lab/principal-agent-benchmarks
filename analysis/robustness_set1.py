"""
Robustness Set 1: pipeline-change rank correlations.

For each of four perturbations applied to the headline scoring pipeline
(no breadth, top-5 reciprocal-rank weights, Claude 3-example prompts):

  1. Top-k cutoff: k = 5 (default) vs k = 3 vs k = 4
  2. Rank weights: 1/k (default) vs uniform 1/5
  3. Judge: Claude (default) vs OpenAI
  4. Per-GWA aggregation: P[A_w ≥ 3.5] / P[H_w ≥ 3.5] (default) vs flat
     mean of A_w / H_w across WORKBank tasks

we recompute (w_auto, w_aug) for the same 317 items and report:

  • Spearman ρ and Kendall τ vs the default pipeline (per perspective)
  • Pareto-frontier overlap (Jaccard) with the default frontier
  • The "% math items dominated by ≥1 general item" Finding-2 statistic

Reads:
  data/judge_loadings.jsonl         — Claude headline judge.
  data/judge_loadings_openai.jsonl  — OpenAI judge variant.
  data/gwa_welfare_scores.csv       — per-GWA p_auto_15 / p_aug_15.
  data/gwa_welfare_scores_mean.csv  — alternative per-GWA aggregation
                                      (flat mean of A_w / H_w rather than
                                      the ≥3.5 proportion).

Writes:
  outputs/robustness_set1.csv
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
WELFARE_CSV = DATA / "gwa_welfare_scores.csv"
WELFARE_MEAN_CSV = DATA / "gwa_welfare_scores_mean.csv"
DEFAULT_OUT = OUT / "robustness_set1.csv"


# ── Family classifier (mirrors analysis/figure3_pareto_ecdf.family) ──────────
MMLU_MATH = {
    "elementary_mathematics", "high_school_mathematics", "college_mathematics",
    "abstract_algebra",
}


def family_of(task: str) -> str:
    if task in ("mathqa", "agieval_aqua_rat", "agieval_sat_math"):
        return "math"
    if task.startswith("mmlu_") and task.removeprefix("mmlu_") in MMLU_MATH:
        return "math"
    return "general"


# ── Aggregation primitives ───────────────────────────────────────────────────

def reciprocal_weights(top_k: int) -> list[float]:
    return [1.0 / k for k in range(1, top_k + 1)]


def uniform_weights(top_k: int) -> list[float]:
    return [1.0 / top_k] * top_k


def score_one(ranked: list[str], gwa_score: dict[str, float],
              weights: list[float]) -> float:
    """w = Σ_{k=1..K} w_k · gwa_score(g_ik), with dropped GWAs (n<5) consuming
    a rank slot but contributing 0. Returns NaN if no kept GWA in top-K."""
    score = 0.0
    used = 0
    for k, gid in enumerate(ranked[: len(weights)]):
        if gid not in gwa_score:
            continue
        score += weights[k] * gwa_score[gid]
        used += 1
    return float("nan") if used == 0 else score


def load_judge_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            rows.append(r)
    return rows


def aggregate(rows: list[dict], gwa_csv: Path, weights: list[float]
              ) -> pd.DataFrame:
    """Per-item w_auto, w_aug from a list of judge rows under given weights."""
    w = pd.read_csv(gwa_csv)
    keep = w[~w["dropped"]]
    auto_g = dict(zip(keep["concept_id"], keep["p_auto_15"]))
    aug_g  = dict(zip(keep["concept_id"], keep["p_aug_15"]))

    by_item: dict[str, dict] = defaultdict(lambda: {
        "lm_eval_task": None, "doc_id": None,
        "w_auto": float("nan"), "w_aug": float("nan"),
    })
    for r in rows:
        d = by_item[r["item_id"]]
        d["lm_eval_task"] = r["lm_eval_task"]
        d["doc_id"] = r["doc_id"]
        ranked = r.get("ranked_gwa_ids") or []
        if r["perspective"] == "automation":
            d["w_auto"] = score_one(ranked, auto_g, weights)
        elif r["perspective"] == "augmentation":
            d["w_aug"] = score_one(ranked, aug_g, weights)

    return pd.DataFrame([
        {"item_id": k, "lm_eval_task": v["lm_eval_task"], "doc_id": v["doc_id"],
         "family": family_of(v["lm_eval_task"]) if v["lm_eval_task"] else "other",
         "w_auto": v["w_auto"], "w_aug": v["w_aug"]}
        for k, v in by_item.items()
    ])


# ── Variant comparison ───────────────────────────────────────────────────────

def pareto_front_ids(df: pd.DataFrame) -> set:
    arr = df[["w_auto", "w_aug"]].values
    out = set()
    for i, (a, b) in enumerate(arr):
        if not ((arr[:, 0] >= a) & (arr[:, 1] >= b)
                & ((arr[:, 0] > a) | (arr[:, 1] > b))).any():
            out.add(df.iloc[i]["item_id"])
    return out


def pct_math_dominated(df: pd.DataFrame) -> float:
    math = df[df["family"] == "math"][["w_auto", "w_aug"]].values
    gen = df[df["family"] == "general"][["w_auto", "w_aug"]].values
    if len(math) == 0:
        return float("nan")
    n_dom = 0
    for a, b in math:
        if ((gen[:, 0] >= a) & (gen[:, 1] >= b)
                & ((gen[:, 0] > a) | (gen[:, 1] > b))).any():
            n_dom += 1
    return 100.0 * n_dom / len(math)


def compare(default_df: pd.DataFrame, variant_df: pd.DataFrame) -> dict[str, float]:
    """Spearman/Kendall on each axis + frontier Jaccard + Finding 2 statistic."""
    merged = default_df[["item_id", "w_auto", "w_aug"]].merge(
        variant_df[["item_id", "w_auto", "w_aug"]],
        on="item_id", suffixes=("_def", "_var"),
    ).dropna()

    spearman_auto, _ = stats.spearmanr(merged["w_auto_def"], merged["w_auto_var"])
    spearman_aug,  _ = stats.spearmanr(merged["w_aug_def"],  merged["w_aug_var"])
    kendall_auto,  _ = stats.kendalltau(merged["w_auto_def"], merged["w_auto_var"])
    kendall_aug,   _ = stats.kendalltau(merged["w_aug_def"],  merged["w_aug_var"])

    f_def = pareto_front_ids(default_df)
    f_var = pareto_front_ids(variant_df)
    jaccard = (len(f_def & f_var) / len(f_def | f_var)) if (f_def or f_var) else float("nan")

    return {
        "n_compared": len(merged),
        "spearman_auto": spearman_auto,
        "spearman_aug":  spearman_aug,
        "kendall_auto":  kendall_auto,
        "kendall_aug":   kendall_aug,
        "frontier_size_default": len(f_def),
        "frontier_size_variant": len(f_var),
        "frontier_overlap":      len(f_def & f_var),
        "frontier_jaccard":      jaccard,
        "pct_math_dominated_default": pct_math_dominated(default_df),
        "pct_math_dominated_variant": pct_math_dominated(variant_df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude_jsonl", type=Path,
                    default=DATA / "judge_loadings.jsonl")
    ap.add_argument("--openai_jsonl", type=Path,
                    default=DATA / "judge_loadings_openai.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Headline / default scores: Claude, k=5, reciprocal weights.
    claude_rows = load_judge_rows(args.claude_jsonl)
    print(f"[default] Claude: {len(claude_rows)} judge rows")
    default_df = aggregate(claude_rows, WELFARE_CSV, reciprocal_weights(5))

    variants: list[tuple[str, pd.DataFrame]] = []
    variants.append(("k=3, recip", aggregate(claude_rows, WELFARE_CSV,
                                             reciprocal_weights(3))))
    variants.append(("k=4, recip", aggregate(claude_rows, WELFARE_CSV,
                                             reciprocal_weights(4))))
    variants.append(("k=5, uniform", aggregate(claude_rows, WELFARE_CSV,
                                               uniform_weights(5))))
    if args.openai_jsonl.exists():
        openai_rows = load_judge_rows(args.openai_jsonl)
        print(f"[variant] OpenAI: {len(openai_rows)} judge rows")
        variants.append(("OpenAI, k=5, recip",
                         aggregate(openai_rows, WELFARE_CSV, reciprocal_weights(5))))
    else:
        print(f"[variant] OpenAI file missing at {args.openai_jsonl} — skipping judge variant.")

    # Variant 4: per-GWA aggregation = flat mean of WORKBank A_w / H_w instead
    # of the default "fraction ≥ 3.5". Same Claude judge, k=5, reciprocal weights.
    if WELFARE_MEAN_CSV.exists():
        variants.append(("GWA mean-agg, k=5, recip",
                         aggregate(claude_rows, WELFARE_MEAN_CSV,
                                   reciprocal_weights(5))))
    else:
        print(f"[variant] mean-agg CSV missing at {WELFARE_MEAN_CSV} — "
              f"skipping per-GWA aggregation variant.")

    rows = []
    for label, var_df in variants:
        stats_dict = compare(default_df, var_df)
        rows.append({"variant": label, **stats_dict})
    df = pd.DataFrame(rows).set_index("variant")

    pd.options.display.float_format = "{:.3f}".format
    cols = ["spearman_auto", "spearman_aug", "kendall_auto", "kendall_aug",
            "frontier_size_default", "frontier_size_variant", "frontier_overlap",
            "frontier_jaccard", "pct_math_dominated_default",
            "pct_math_dominated_variant"]
    print("\n" + "=" * 80)
    print("Set 1: pipeline-change robustness")
    print("=" * 80)
    print(df[cols].to_string())

    df.to_csv(args.out)
    print(f"\n[csv] wrote {args.out}")


if __name__ == "__main__":
    main()

"""
Robustness Set 2: human-rater check on the 28 held-out items.

Compute the headline welfare scoring (no breadth, top-5 reciprocal weights)
for each of three raters — A1, A2, claude3ex — using their top-5 GWA rankings
on the same 28 items, then report:

  • Finding 1 (loading mass per quadrant): for each rater × perspective, the
    sum of top-5 slot mass that lands in each of {Green-Light, Augment-Only,
    Automate-Only, Red-Light}, with the threshold split at p=3.0 on the
    bare-share (rescaled to [1, 5]) per-GWA welfare scores.

  • Finding 2 (math vs general): per-axis mean welfare gap, Pareto-frontier
    composition, % math items dominated by ≥1 general item.

Reads:
  data/heldout_raters/heldout_28_items.json
  data/heldout_raters/family_map.json
  data/heldout_raters/{A1,A2,claude3ex}_{automation,augmentation}.jsonl
  data/gwa_welfare_scores.csv

Writes:
  outputs/robustness_set2_finding1_loading_mass.csv
  outputs/robustness_set2_finding2_stats.csv
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RATER_DIR = DATA / "heldout_raters"
WELFARE_CSV = DATA / "gwa_welfare_scores.csv"
OUT = ROOT / "outputs"

THRESHOLD = 3.0
TOP_K = 5
RATERS = ("A1", "A2", "claude3ex")


# ── Aggregation: same as score_items_no_breadth, with parameterised k & wts ──

def score_one(ranked: list[str], gwa_score: dict[str, float],
              weights: list[float]) -> float:
    score = 0.0
    used = 0
    for k, gid in enumerate(ranked[: len(weights)]):
        if gid not in gwa_score:
            continue
        score += weights[k] * gwa_score[gid]
        used += 1
    return float("nan") if used == 0 else score


def reciprocal_weights(top_k: int) -> list[float]:
    return [1.0 / k for k in range(1, top_k + 1)]


# ── Finding 1: loading mass per quadrant ─────────────────────────────────────

def quadrant_of(p_auto: float, p_aug: float, threshold: float = THRESHOLD) -> str:
    if p_auto >= threshold and p_aug >= threshold: return "green"
    if p_auto >= threshold and p_aug <  threshold: return "automate"
    if p_auto <  threshold and p_aug >= threshold: return "augment"
    return "low"


def loading_mass(rows: list[dict], gwa_csv: Path,
                 perspective: str, threshold: float = THRESHOLD,
                 top_k: int = TOP_K) -> dict[str, float]:
    """For each item, distribute its top-k slots across the four GWA quadrants.
    Returns the *fraction of slots* in each quadrant — pooled across items,
    normalised so green+automate+augment+low sums to 100. Slots whose GWA is
    dropped (n<5 in WORKBank) are excluded.
    """
    g = pd.read_csv(gwa_csv)
    g = g[~g["dropped"]]
    g["quadrant"] = g.apply(
        lambda r: quadrant_of(r["p_auto_15"], r["p_aug_15"], threshold),
        axis=1,
    )
    quadrant_of_gwa: dict[str, str] = dict(zip(g["concept_id"], g["quadrant"]))

    mass = {q: 0.0 for q in ("green", "automate", "augment", "low")}
    n_items = 0
    n_slots_total = 0
    for r in rows:
        if r["perspective"] != perspective:
            continue
        n_items += 1
        for gid in (r.get("ranked_gwa_ids") or [])[:top_k]:
            q = quadrant_of_gwa.get(gid)
            if q is None:
                continue
            mass[q] += 1.0
            n_slots_total += 1
    if n_slots_total > 0:
        mass = {q: 100.0 * v / n_slots_total for q, v in mass.items()}
    mass["n_items"] = n_items
    return mass


# ── Finding 2: math vs general welfare gap, frontier, dominance ──────────────

def aggregate_one_rater(auto_path: Path, aug_path: Path, gwa_csv: Path,
                        family: dict[str, str], top_k: int = TOP_K
                        ) -> pd.DataFrame:
    weights = reciprocal_weights(top_k)
    w = pd.read_csv(gwa_csv)
    keep = w[~w["dropped"]]
    auto_g = dict(zip(keep["concept_id"], keep["p_auto_15"]))
    aug_g  = dict(zip(keep["concept_id"], keep["p_aug_15"]))

    by_item: dict[str, dict] = defaultdict(lambda: {
        "w_auto": float("nan"), "w_aug": float("nan"),
    })
    for r in (json.loads(l) for l in auto_path.open()):
        by_item[r["task_id"]]["w_auto"] = score_one(r["ranked_gwa_ids"], auto_g, weights)
    for r in (json.loads(l) for l in aug_path.open()):
        by_item[r["task_id"]]["w_aug"] = score_one(r["ranked_gwa_ids"], aug_g, weights)

    rows = []
    for tid, d in by_item.items():
        rows.append({
            "task_id": tid,
            "family": family.get(tid, "other"),
            "w_auto": d["w_auto"], "w_aug": d["w_aug"],
        })
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    arr = df[["w_auto", "w_aug"]].values
    on = np.ones(len(arr), dtype=bool)
    for i, (a, b) in enumerate(arr):
        if ((arr[:, 0] >= a) & (arr[:, 1] >= b)
                & ((arr[:, 0] > a) | (arr[:, 1] > b))).any():
            on[i] = False
    return df[on].copy()


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


def finding2_stats_for(df: pd.DataFrame) -> dict[str, float]:
    df = df[df["family"].isin({"math", "general"})].dropna(subset=["w_auto", "w_aug"])
    n_math = (df["family"] == "math").sum()
    n_gen = (df["family"] == "general").sum()
    front = pareto_front(df)
    n_front_math = (front["family"] == "math").sum()
    n_front_gen = (front["family"] == "general").sum()
    return {
        "n_math": int(n_math),
        "n_gen": int(n_gen),
        "math_mean_w_auto":     df.loc[df["family"] == "math",    "w_auto"].mean(),
        "gen_mean_w_auto":      df.loc[df["family"] == "general", "w_auto"].mean(),
        "diff_w_auto":          df.loc[df["family"] == "general", "w_auto"].mean()
                                 - df.loc[df["family"] == "math",    "w_auto"].mean(),
        "math_mean_w_aug":      df.loc[df["family"] == "math",    "w_aug"].mean(),
        "gen_mean_w_aug":       df.loc[df["family"] == "general", "w_aug"].mean(),
        "diff_w_aug":           df.loc[df["family"] == "general", "w_aug"].mean()
                                 - df.loc[df["family"] == "math",    "w_aug"].mean(),
        "frontier_n":           int(len(front)),
        "frontier_n_math":      int(n_front_math),
        "frontier_n_general":   int(n_front_gen),
        "pct_math_dominated":   pct_math_dominated(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater_dir", type=Path, default=RATER_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out_finding1", type=Path,
                    default=OUT / "robustness_set2_finding1_loading_mass.csv")
    ap.add_argument("--out_finding2", type=Path,
                    default=OUT / "robustness_set2_finding2_stats.csv")
    args = ap.parse_args()

    args.out_finding1.parent.mkdir(parents=True, exist_ok=True)

    held = json.loads((args.rater_dir / "heldout_28_items.json").read_text())
    family = json.loads((args.rater_dir / "family_map.json").read_text())
    print(f"[set2] {len(held)} held-out items")
    fam_counts = pd.Series([family.get(i, "other") for i in held]).value_counts()
    print(f"[set2] family breakdown: {dict(fam_counts)}")

    f1_rows = []
    for rater in RATERS:
        for persp in ("automation", "augmentation"):
            path = args.rater_dir / f"{rater}_{persp}.jsonl"
            rows = [json.loads(l) for l in path.open()]
            mass = loading_mass(rows, WELFARE_CSV, persp, args.threshold)
            f1_rows.append({"rater": rater, "perspective": persp, **mass})
    f1 = pd.DataFrame(f1_rows)
    cols = ["rater", "perspective", "n_items", "green", "automate", "augment", "low"]
    f1 = f1[cols]
    f1.to_csv(args.out_finding1, index=False)
    pd.options.display.float_format = "{:.3f}".format
    print("\n" + "=" * 72)
    print(f"Set 2 — Finding 1: % of top-5 slot mass per quadrant "
          f"(threshold={args.threshold})")
    print("=" * 72)
    print(f1.to_string(index=False))
    print(f"\n[csv] wrote {args.out_finding1}")

    f2_rows = []
    for rater in RATERS:
        auto_p = args.rater_dir / f"{rater}_automation.jsonl"
        aug_p  = args.rater_dir / f"{rater}_augmentation.jsonl"
        df = aggregate_one_rater(auto_p, aug_p, WELFARE_CSV, family)
        s = finding2_stats_for(df)
        f2_rows.append({"rater": rater, **s})
    f2 = pd.DataFrame(f2_rows)
    f2.to_csv(args.out_finding2, index=False)
    print("\n" + "=" * 72)
    print("Set 2 — Finding 2: math vs general welfare gap")
    print("=" * 72)
    cols = ["rater", "n_math", "n_gen",
            "math_mean_w_auto", "gen_mean_w_auto", "diff_w_auto",
            "math_mean_w_aug",  "gen_mean_w_aug",  "diff_w_aug",
            "frontier_n", "frontier_n_math", "frontier_n_general",
            "pct_math_dominated"]
    print(f2[cols].to_string(index=False))
    print(f"\n[csv] wrote {args.out_finding2}")


if __name__ == "__main__":
    main()

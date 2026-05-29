"""
Build the intermediate per-item CSVs that the analysis scripts read.

This script is only needed if you are regenerating data/ from raw evaluation
panels (e.g. after a methodology change). The shipped data/*.csv are already
the output of this script — running `make all` does not require it.

Inputs (relative to $LFS_BASE):
  cost_noise_4B/items_master.jsonl         — one row per OLMES item.
  cost_noise_4B/items_with_rank_ci.jsonl   — same rows + bootstrap rank CIs.
  welfare_judge/item_welfare_scores.jsonl  — judge welfare scores.
  welfare_judge/gwa_1pct.jsonl             — judge top-k GWA loadings.
  cost_noise_summaries.jsonl               — cached LLM topic summaries.

Outputs (written to data/):
  noise.csv             — items with keeps_noise == True, with rank CI cols.
  cost.csv              — items with keeps_cost == True, with rank CI cols.
  welfare.csv           — welfare-judged items, per-judge P_auto / P_aug.
  judge_loadings.csv    — per (item, judge, perspective) top-k GWA list.
  item_topics.csv       — short LLM topic summary per (task, doc_id).

Run:
  LFS_BASE=/path/to/lfs python -m analysis.build_intermediate_csvs
"""

import json
import os
from pathlib import Path

import pandas as pd

LFS_BASE = Path(os.environ["LFS_BASE"]) if "LFS_BASE" in os.environ else None
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

NOISE_COLS = [
    "task", "doc_id",
    "noise_sigma_hat", "noise_sigma2", "noise_mean_phat", "noise_n_obs",
    "subgroup",
    "rank_headline", "rank_lo", "rank_hi", "rank_p50",
]

COST_COLS = [
    "task", "doc_id",
    "anchor_phat",
    "beta_pt", "beta_sft", "beta_rl",
    "cost_pt_usd", "cost_sft_usd", "cost_rl_usd",
    "cost_min_usd", "cheapest_axis",
    "axis_pt_improving", "axis_sft_improving", "axis_rl_improving",
    "rank_headline", "rank_lo", "rank_hi", "rank_p50",
]


def load_master(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def assemble_noise(master: list[dict], rank_ci: dict[tuple, dict]) -> pd.DataFrame:
    out = []
    for r in master:
        if not r.get("keeps_noise"):
            continue
        ci = rank_ci.get((r["task"], r["doc_id"]), {})
        out.append({
            "task": r["task"],
            "doc_id": r["doc_id"],
            "noise_sigma_hat": r["noise_sigma"],
            "noise_sigma2": r["noise_sigma2"],
            "noise_mean_phat": r["noise_mean_phat"],
            "noise_n_obs": r["noise_n_obs"],
            "subgroup": r["subgroup"],
            "rank_headline": ci.get("noise_rank_headline"),
            "rank_lo": ci.get("noise_rank_lo"),
            "rank_hi": ci.get("noise_rank_hi"),
            "rank_p50": ci.get("noise_rank_p50"),
        })
    return pd.DataFrame(out, columns=NOISE_COLS)


def assemble_cost(master: list[dict], rank_ci: dict[tuple, dict]) -> pd.DataFrame:
    """Cost-eligible items: paper's headline cost ranking universe.

    Filter mirrors `scripts/bootstrap_rank_ci.py`:
      keeps_both AND at least one improving axis (β̂ > 0).
    The resulting ~14,908 items are the universe of Table 1 ranks."""
    out = []
    for r in master:
        if not r.get("keeps_both"):
            continue
        if not (r.get("beta_PT", 0) > 0 or r.get("beta_SFT", 0) > 0
                or r.get("beta_RL", 0) > 0):
            continue
        ci = rank_ci.get((r["task"], r["doc_id"]), {})
        out.append({
            "task": r["task"],
            "doc_id": r["doc_id"],
            "anchor_phat": r["anchor_phat"],
            "beta_pt": r["beta_PT"],
            "beta_sft": r["beta_SFT"],
            "beta_rl": r["beta_RL"],
            "cost_pt_usd": r["cost_PT_$"],
            "cost_sft_usd": r["cost_SFT_$"],
            "cost_rl_usd": r["cost_RL_$"],
            "cost_min_usd": r["cost_min_$"],
            "cheapest_axis": r["cheapest_axis"],
            "axis_pt_improving": r["axis_PT_improving"],
            "axis_sft_improving": r["axis_SFT_improving"],
            "axis_rl_improving": r["axis_RL_improving"],
            "rank_headline": ci.get("rank_headline"),
            "rank_lo": ci.get("rank_lo"),
            "rank_hi": ci.get("rank_hi"),
            "rank_p50": ci.get("rank_p50"),
        })
    return pd.DataFrame(out, columns=COST_COLS)


def load_rank_ci(path: Path) -> dict[tuple, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["task"], r["doc_id"])] = r
    return out


def assemble_topics(path: Path) -> pd.DataFrame:
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out.append({"task": r["task"], "doc_id": r["doc_id"],
                        "topic": r["summary"]})
    return pd.DataFrame(out)


def assemble_judge_loadings(path: Path) -> pd.DataFrame:
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            ranked = r.get("ranked_gwa_ids") or []
            out.append({
                "task": r["lm_eval_task"],
                "doc_id": r["doc_id"],
                "judge": r["model"],
                "perspective": r["perspective"],
                "ranked_gwa_ids": ";".join(ranked),
            })
    return pd.DataFrame(out)


def assemble_welfare(path: Path) -> pd.DataFrame:
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            per = r.get("P_auto_per_model", {})
            per_aug = r.get("P_aug_per_model", {})
            out.append({
                "task": r["lm_eval_task"],
                "doc_id": r["doc_id"],
                "P_auto_claude": per.get("claude"),
                "P_aug_claude":  per_aug.get("claude"),
                "P_auto_openai": per.get("openai_reasoning"),
                "P_aug_openai":  per_aug.get("openai_reasoning"),
            })
    return pd.DataFrame(out)


def main() -> None:
    if LFS_BASE is None:
        raise SystemExit("Set LFS_BASE to the directory holding the raw cost/noise/welfare panels.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    master_path = LFS_BASE / "cost_noise_4B" / "items_master.jsonl"
    rank_ci_path = LFS_BASE / "cost_noise_4B" / "items_with_rank_ci.jsonl"
    welfare_path = LFS_BASE / "welfare_judge" / "item_welfare_scores.jsonl"
    loadings_path = LFS_BASE / "welfare_judge" / "gwa_1pct.jsonl"
    topics_path = LFS_BASE / "cost_noise_summaries.jsonl"
    for p in (master_path, rank_ci_path, welfare_path, loadings_path, topics_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    master = load_master(master_path)
    rank_ci = load_rank_ci(rank_ci_path)

    noise = assemble_noise(master, rank_ci)
    cost = assemble_cost(master, rank_ci)
    welfare = assemble_welfare(welfare_path)
    loadings = assemble_judge_loadings(loadings_path)
    topics = assemble_topics(topics_path)

    noise.to_csv(OUT_DIR / "noise.csv", index=False)
    cost.to_csv(OUT_DIR / "cost.csv", index=False)
    welfare.to_csv(OUT_DIR / "welfare.csv", index=False)
    loadings.to_csv(OUT_DIR / "judge_loadings.csv", index=False)
    topics.to_csv(OUT_DIR / "item_topics.csv", index=False)

    print(f"wrote {OUT_DIR / 'noise.csv'}            rows={len(noise):>6,}")
    print(f"wrote {OUT_DIR / 'cost.csv'}             rows={len(cost):>6,}")
    print(f"wrote {OUT_DIR / 'welfare.csv'}          rows={len(welfare):>6,}")
    print(f"wrote {OUT_DIR / 'judge_loadings.csv'}   rows={len(loadings):>6,}")
    print(f"wrote {OUT_DIR / 'item_topics.csv'}      rows={len(topics):>6,}")


if __name__ == "__main__":
    main()

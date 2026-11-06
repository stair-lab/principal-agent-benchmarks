"""
Build the intermediate per-item CSVs that the analysis scripts read.

The shipped data/*.csv are already the output of this script — running
`make all` does not require it. Re-run it after replacing
data/llm-judge/judge_loadings.jsonl (e.g. with the output of
`analysis.refresh_judge`) or after regenerating the raw evaluation panels.

welfare.csv is always rebuilt, from shipped inputs only:
  data/llm-judge/judge_loadings.jsonl     — judge top-5 GWA ranking per (item, perspective).
  data/gwa-scores/gwa_welfare_scores.csv  — per-GWA p_auto_15 / p_aug_15.

The remaining CSVs are rebuilt only when $LFS_BASE is set. Inputs (relative to $LFS_BASE):
  cost_noise_4B/items_master.jsonl         — one row per OLMES item.
  cost_noise_4B/items_with_rank_ci.jsonl   — same rows + bootstrap rank CIs.
  cost_noise_summaries.jsonl               — cached LLM topic summaries.

Outputs (written under data/):
  item-level-evals/welfare.csv  — per-item w_auto / w_aug (see data/README.md).
  item-level-evals/noise.csv    — items with keeps_noise == True, with rank CI cols.
  item-level-evals/cost.csv     — items with keeps_cost == True, with rank CI cols.
  llm-judge/item_topics.csv     — short LLM topic summary per (task, doc_id).

Run:
  python -m analysis.build_intermediate_csvs                          # welfare.csv only
  LFS_BASE=/path/to/lfs python -m analysis.build_intermediate_csvs    # everything
"""

import json
import os
from pathlib import Path

import pandas as pd

from analysis.robustness_set1 import reciprocal_weights, score_one

LFS_BASE = Path(os.environ["LFS_BASE"]) if "LFS_BASE" in os.environ else None
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
JUDGE_LOADINGS = OUT_DIR / "llm-judge" / "judge_loadings.jsonl"
GWA_SCORES = OUT_DIR / "gwa-scores" / "gwa_welfare_scores.csv"
TOP_K = 5

# dataset_family column of welfare.csv for the 317 headline items: MMLU subject
# categories, with the MMLU math subjects grouped with the other math benchmarks.
# Extend this when judging items from tasks not listed here.
DATASET_FAMILY = {
    "math": [
        "agieval_aqua_rat", "agieval_sat_math", "mathqa",
        "mmlu_college_mathematics", "mmlu_elementary_mathematics",
        "mmlu_high_school_mathematics",
    ],
    "mmlu_stem": [
        "mmlu_anatomy", "mmlu_astronomy", "mmlu_college_chemistry",
        "mmlu_college_physics", "mmlu_conceptual_physics",
        "mmlu_electrical_engineering", "mmlu_high_school_chemistry",
        "mmlu_high_school_physics", "mmlu_machine_learning",
        "mmlu_medical_genetics",
    ],
    "mmlu_humanities": [
        "mmlu_formal_logic", "mmlu_high_school_european_history",
        "mmlu_high_school_us_history", "mmlu_high_school_world_history",
        "mmlu_international_law", "mmlu_jurisprudence",
        "mmlu_logical_fallacies", "mmlu_moral_disputes", "mmlu_moral_scenarios",
        "mmlu_philosophy", "mmlu_prehistory", "mmlu_professional_law",
    ],
    "mmlu_social": [
        "mmlu_econometrics", "mmlu_high_school_geography",
        "mmlu_high_school_government_and_politics",
        "mmlu_high_school_macroeconomics", "mmlu_high_school_microeconomics",
        "mmlu_high_school_psychology", "mmlu_human_sexuality",
        "mmlu_professional_psychology", "mmlu_security_studies",
        "mmlu_sociology",
    ],
    "mmlu_other": [
        "mmlu_business_ethics", "mmlu_clinical_knowledge",
        "mmlu_college_medicine", "mmlu_global_facts", "mmlu_human_aging",
        "mmlu_management", "mmlu_miscellaneous", "mmlu_nutrition",
        "mmlu_professional_accounting", "mmlu_professional_medicine",
        "mmlu_virology",
    ],
    "reasoning": [
        "arc_challenge", "commonsense_qa",
    ],
    "pretrain": [
        "hellaswag", "piqa",
    ],
}
TASK_FAMILY = {t: fam for fam, tasks in DATASET_FAMILY.items() for t in tasks}

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


def assemble_welfare(loadings_path: Path, gwa_path: Path) -> pd.DataFrame:
    """Per-item w_auto / w_aug: reciprocal-rank (1/k) sum over the judge's
    top-5 GWAs of p_auto_15 / p_aug_15. GWAs dropped for n<5 in WORKBank
    consume a rank slot but contribute 0."""
    g = pd.read_csv(gwa_path)
    keep = g[~g["dropped"]]
    gwa_score = {
        "automation": dict(zip(keep["concept_id"], keep["p_auto_15"])),
        "augmentation": dict(zip(keep["concept_id"], keep["p_aug_15"])),
    }
    col = {"automation": "w_auto", "augmentation": "w_aug"}
    weights = reciprocal_weights(TOP_K)

    items: dict[tuple, dict] = {}
    with loadings_path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            # judge_loadings.jsonl uses lm_eval_task; refresh_judge output uses task.
            task = r.get("lm_eval_task", r.get("task"))
            if task not in TASK_FAMILY:
                raise SystemExit(f"no dataset_family for task {task!r}; add it to DATASET_FAMILY")
            d = items.setdefault((task, r["doc_id"]), {
                "task": task, "doc_id": r["doc_id"],
                "dataset_family": TASK_FAMILY[task],
                "w_auto": float("nan"), "w_aug": float("nan"),
            })
            d[col[r["perspective"]]] = score_one(
                r.get("ranked_gwa_ids") or [], gwa_score[r["perspective"]], weights)
    return pd.DataFrame(list(items.values()),
                        columns=["task", "doc_id", "dataset_family", "w_auto", "w_aug"])


def main() -> None:
    for sub in ("item-level-evals", "llm-judge"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    welfare = assemble_welfare(JUDGE_LOADINGS, GWA_SCORES)
    welfare.to_csv(OUT_DIR / "item-level-evals" / "welfare.csv", index=False,
                   float_format="%.6f")
    print(f"wrote {OUT_DIR / 'item-level-evals' / 'welfare.csv'}          rows={len(welfare):>6,}")

    if LFS_BASE is None:
        print("LFS_BASE not set; skipping noise.csv, cost.csv, item_topics.csv.")
        return

    master_path = LFS_BASE / "cost_noise_4B" / "items_master.jsonl"
    rank_ci_path = LFS_BASE / "cost_noise_4B" / "items_with_rank_ci.jsonl"
    topics_path = LFS_BASE / "cost_noise_summaries.jsonl"
    for p in (master_path, rank_ci_path, topics_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    master = load_master(master_path)
    rank_ci = load_rank_ci(rank_ci_path)

    noise = assemble_noise(master, rank_ci)
    cost = assemble_cost(master, rank_ci)
    topics = assemble_topics(topics_path)

    noise.to_csv(OUT_DIR / "item-level-evals" / "noise.csv", index=False)
    cost.to_csv(OUT_DIR / "item-level-evals" / "cost.csv", index=False)
    topics.to_csv(OUT_DIR / "llm-judge" / "item_topics.csv", index=False)

    print(f"wrote {OUT_DIR / 'item-level-evals' / 'noise.csv'}            rows={len(noise):>6,}")
    print(f"wrote {OUT_DIR / 'item-level-evals' / 'cost.csv'}             rows={len(cost):>6,}")
    print(f"wrote {OUT_DIR / 'llm-judge' / 'item_topics.csv'}      rows={len(topics):>6,}")


if __name__ == "__main__":
    main()

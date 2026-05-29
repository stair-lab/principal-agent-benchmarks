# Per-item primary outputs

Each CSV is the input to the corresponding paper finding. All are keyed by
`(task, doc_id)` unless stated.

## `noise.csv`  (Finding 4, Tables 2 and 4)

Items passing the noise eligibility filter on the PolyPythias 410M panel:
`p_bar ∈ [0.10, 0.90]` and `noise_sigma2 ≥ 1e-6`. Pooled within-checkpoint
across-seed variance over 8 seeds × 10 log-spaced checkpoints (N=80, divisor 70).

| Column | Description |
|---|---|
| `task` | OLMES task id (e.g. `mmlu_high_school_world_history`). |
| `doc_id` | Item index within the task. |
| `noise_sigma_hat` | √(pooled variance). |
| `noise_sigma2` | Pooled across-seed variance. |
| `noise_mean_phat` | Mean p̂ across all (seed, checkpoint) pairs. |
| `noise_n_obs` | N = S·K observations contributing. |
| `subgroup` | `math` / `non-math`. |

## `cost.csv`  (Finding 3, Table 1)

Items passing the cost eligibility filter on the EvoLM 4B suite: anchor
`p̂ ∈ [0.05, 0.95]` and at least one strictly improving axis (β̂ > 0).
Three 1D OLS fits per item along PT, SFT, and RL grids.

| Column | Description |
|---|---|
| `task`, `doc_id` | As above. |
| `anchor_phat` | p̂ at the 4B anchor (160BT PT, FW8FM42 CPT, 100k SFT × 1ep, no RL). |
| `beta_pt`, `beta_sft`, `beta_rl` | Per-item OLS slope per native unit (token, sample, prompt). |
| `cost_pt_usd`, `cost_sft_usd`, `cost_rl_usd` | Dollar cost per 0.10 lift; NaN where slope ≤ 0. |
| `cost_min_usd` | Cost via the cheapest improving axis. |
| `cheapest_axis` | One of `PT`, `SFT`, `RL`. |
| `axis_pt_improving`, `axis_sft_improving`, `axis_rl_improving` | Bool: β̂ > 0 on that axis. |

## `welfare.csv`  (Finding 2, Figure 3)

OLMES headline subset (n=317) for which the LLM-judge pipeline ran. Each row
is one (task, doc_id) item; welfare scores aggregate the judge's top-5 GWA
ranking against `gwa_welfare_scores.csv` via reciprocal-rank weighting (no
breadth weighting).

For each item i,

    w_auto(i) = Σ_{k=1..5} (1/k) · p_auto_15(g_ik)
    w_aug(i)  = Σ_{k=1..5} (1/k) · p_aug_15(g_ik)

where g_ik is the k-th GWA in the judge's ranking and `p_X_15` is the per-GWA
WORKBank share rescaled to [1, 5]. GWAs with n<5 in WORKBank are dropped:
they consume a rank slot but contribute 0.

| Column | Description |
|---|---|
| `task`, `doc_id` | As above. |
| `dataset_family` | `math`, `mmlu_stem`, `mmlu_humanities`, `mmlu_social`, `mmlu_other`, `reasoning`, or `pretrain`. |
| `w_auto`, `w_aug` | Per-item welfare scores in raw (un-rescaled) units; the figure scripts rescale to [1, 5] using fixed theoretical bounds. |

## `gwa_welfare_scores.csv`  (input to Figure 2)

24 retained O*NET Generalized Work Activities (3 dropped for n<5 in WORKBank).

| Column | Description |
|---|---|
| `gwa` | GWA name. |
| `concept_id` | Stable id used downstream. |
| `p_auto`, `p_aug` | P[A_w ≥ 3.5] and P[H_w ≥ 3.5] per WORKBank. |
| `p_auto_15`, `p_aug_15` | Per-axis rescaling of `p_auto`/`p_aug` to [1, 5]; this is what the no-breadth figures use. |
| `breadth_raw`, `log_breadth`, `breadth` | Diagnostic columns from the older breadth-weighted aggregation; kept for reference but not used by `make all`. |
| `P_auto`, `P_aug` | (p_X · breadth) jointly rescaled to [1, 5]; legacy, not used. |

## `gwa_welfare_scores_mean.csv`  (Robustness Set 1)

Same 24 retained GWAs and same column names as `gwa_welfare_scores.csv`,
but `p_auto_15`/`p_aug_15` here hold a different per-GWA aggregation: the
flat mean of WORKBank A_w / H_w ratings across tasks (rather than the
"fraction ≥ 3.5" proportion). Used as the alternative scoring in
`make robustness-set1`; not consumed by the headline figures.

## `judge_loadings.jsonl`  (input to Figure 2)

One row per (item, perspective) capturing the Claude judge's top-5 GWA
ranking for that item.

| Field | Description |
|---|---|
| `lm_eval_task`, `doc_id` | OLMES task id and item index. |
| `item_id` | `<task>_<doc_id:06d>`. |
| `perspective` | `automation` or `augmentation`. |
| `model` | Always `claude` (Claude Opus 4.5 in the headline run). |
| `ranked_gwa_ids` | List of 3–10 GWA ids in rank order. |
| `sub_steps` | Judge's enumeration of solution sub-steps (audit trail). |
| `rationale` | Free-text rationale from the judge. |
| `ok` | True iff the judge call returned a valid response. |

## `judge_loadings_openai.jsonl`  (Robustness Set 1)

Same schema as `judge_loadings.jsonl` but produced by the OpenAI reasoning
judge instead of Claude. 634 rows (317 items × 2 perspectives). Used by
`make robustness-set1` as the judge-variant comparison; not consumed by
the headline figures.

## `heldout_raters/`  (Robustness Set 2)

28 items judged independently by three raters (two anonymised humans plus
Claude-with-3-examples) for the human-rater check.

| File | Description |
|---|---|
| `heldout_28_items.json` | Ordered list of 28 opaque task ids. |
| `family_map.json` | task_id → `math` / `general`, pre-computed from the labelling-package metadata. |
| `{A1,A2,claude3ex}_{automation,augmentation}.jsonl` | Per-rater × perspective ranked top-5 GWAs; one row per item with `task_id`, `perspective`, `ranked_gwa_ids`, `rater`. |

## `item_topics.csv`  (Tables 1, 2)

Short LLM-generated topic phrases for items appearing in the paper's quantile
tables. Provides the `Topic` column without exposing item prompts.

| Column | Description |
|---|---|
| `task`, `doc_id` | As above. |
| `topic` | One-line phrase summarizing the item (e.g. `calculating buoyancy`). |

## `item_margins.parquet`  (Table 4)

Per-item median gold-vs-runner-up log-prob margin across the 80 (seed × ckpt)
observations on the 410M PolyPythia panel. Joined with σ̂ for the noise-decomposition
table.

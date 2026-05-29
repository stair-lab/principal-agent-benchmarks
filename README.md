# principal-agent-benchmarks

Code and per-item data to reproduce all figures and tables in:

> **Welfare, Improvability, and Noise: A Principal–Agent Theory of Optimal
> Benchmark Item Aggregation**
> Andreas Haupt, Justin Hartenstein, Anka Reuel, Mykel J. Kochenderfer,
> Sanmi Koyejo.

## Layout

```
.
├── data/        Per-item primary outputs (CSV) — sufficient for all figures/tables.
├── analysis/    One script per paper artifact; reads only data/.
│                Plus build_intermediate_csvs.py (regenerates data/),
│                fetch_olmes_items.py (materializes items.jsonl), and
│                refresh_judge.py (re-runs the LLM-as-judge).
├── env.example  Copy to .env and fill in if you run refresh_judge.
├── LICENSE      MIT.
└── Makefile     Reproduction entry point.
```

## Reproduce all paper artifacts

```bash
pip install -e .
make all     # writes outputs/figureN.{pdf,png}, outputs/tableN.tex
```

`make all` runs only `analysis/` — the input is the shipped `data/*.csv` and
`data/item_margins.parquet`. It does not require GPU access, the raw evaluation
panels, or LLM API keys. Reproduces Figures 2 and 3, Tables 1, 2, and 4.

## Data files in `data/`

| File | Rows | Description |
|---|---:|---|
| `noise.csv` | 20,206 | Per-item across-seed noise on PolyPythias 410M (8 seeds × 10 checkpoints), with bootstrap rank CIs. |
| `cost.csv`  | 14,908 | Per-item OLS slopes and dollar costs on EvoLM 4B (PT/SFT/RL grids), with bootstrap rank CIs. |
| `welfare.csv` | 317 | Per-item welfare scores `w_auto`, `w_aug` from no-breadth aggregation of LLM-judge top-5 GWA loadings × WORKBank shares. |
| `gwa_welfare_scores.csv` | 27 | Per-GWA p_auto/p_aug (and rescaled p_auto_15/p_aug_15 used by Figure 2). 24 retained, 3 dropped for n<5. |
| `gwa_welfare_scores_mean.csv` | 27 | Alternative per-GWA aggregation (flat mean of A_w / H_w); input to `make robustness-set1`. |
| `judge_loadings.jsonl` | 634 | One row per (item, perspective) — Claude's top-5 GWA ranking with rationale. |
| `judge_loadings_openai.jsonl` | 634 | OpenAI judge variant; input to `make robustness-set1`. |
| `heldout_raters/` | 8 files | 28 held-out items × 3 raters (A1, A2, claude3ex) × 2 perspectives; input to `make robustness-set2`. |
| `item_topics.csv` | 105 | Short LLM-generated topic phrases for items appearing in the paper's quantile tables. |
| `item_margins.parquet` | 19,999 | Per-item gold-vs-runner-up log-prob margins on the 410M panel (input to Table 4). |

Each is keyed by `(task, doc_id)` (and `gwa` / `judge` / `perspective` where
applicable). Schema details are in `data/README.md`.

To look up the prompt or choices for a given `(task, doc_id)` pair, fetch the
task via `lm-evaluation-harness` (the OLMES-aligned task definitions index by
the same `doc_id`).

## Robustness checks

Two independent robustness checks on the welfare findings:

```bash
make robustness          # both
make robustness-set1     # pipeline perturbations
make robustness-set2     # human-rater check
```

**Set 1** — pipeline perturbations on the headline 317-item set. Compares
the default (Claude, top-5, reciprocal-rank weights, "fraction ≥ 3.5"
per-GWA aggregation) against: top-3 / top-4 cutoffs, uniform weights,
the OpenAI judge, and a flat-mean per-GWA aggregation.
Reports Spearman ρ, Kendall τ, Pareto-frontier Jaccard overlap, and the
Finding-2 dominance statistic per variant. Output: `outputs/robustness_set1.csv`.

**Set 2** — human-rater check on a 28-item held-out set. Three raters (A1,
A2, claude3ex) independently produced top-5 GWA rankings; we score each
rater's items under the headline aggregation and compare to the headline
findings. Outputs: `outputs/robustness_set2_finding1_loading_mass.csv`
and `outputs/robustness_set2_finding2_stats.csv`.

Neither requires API access or GPU; both run in <30 seconds on the
shipped data.

## Re-running the LLM-as-judge

`make refresh-judge` re-runs the GWA loadings against current LLMs. **Costs
real money** (Claude / GPT API calls); defaults to a one-call smoke test so
you can verify your setup before committing to the full run.

```bash
pip install -e .[refresh-judge]
cp env.example .env && edit .env  # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY
source .env

# 1. Materialize prompts + choices for the items in welfare.csv.
#    Reads OLMES task definitions from lm-evaluation-harness; not part of
#    this paper's contributions.
python -m analysis.fetch_olmes_items

# 2. Re-run the judge.
make refresh-judge                                # 1 call, smoke test
make refresh-judge JUDGE=claude N_ITEMS=10        # 10 items × 2 perspectives
make refresh-judge JUDGE=openai PERSPECTIVE=automation N_ITEMS=50
python -m analysis.refresh_judge --all            # full re-run
```

The output is written to `data/judge_loadings_refreshed.jsonl`; replace
`data/judge_loadings.csv` with it (and re-run `analysis/build_intermediate_csvs.py`)
to flow the new loadings through to figures and tables.

## Re-running the upstream chain end-to-end

`analysis/build_intermediate_csvs.py` rebuilds `data/*.csv` from raw evaluation
panels. It is **not** required for `make all`. Required inputs (not shipped):

- **PolyPythias 410M** (`EleutherAI/pythia-410m-seed{1..8}`) — OLMES per-item
  logprobs, 8 seeds × 10 checkpoints. → `noise.csv`.
- **EvoLM 4B suite** — OLMES per-item logprobs along three intervention grids
  (3 PT × 9 SFT × 9 RL). → `cost.csv`.
- **WORKBank** worker-preference data (Shao et al., 2025). Released under
  no declared license; fetch from the upstream
  [SALT-NLP/workbank](https://github.com/SALT-NLP/workbank) repo and place
  `task_level_aw_hw.csv` and `task_statement_with_metadata.csv` under
  `data/workbank/`. → input to the per-GWA welfare scores.
- **OLMES item set** — fetched on demand by `analysis/fetch_olmes_items.py`
  via `lm-evaluation-harness` (see above).

Set `LFS_BASE` to the directory holding the raw panels, then run
`python -m analysis.build_intermediate_csvs`.

## Citation

```bibtex
@misc{haupt2026welfare,
  title  = {Welfare, Improvability, and Noise: A Principal--Agent Theory of Optimal Benchmark Item Aggregation},
  author = {Haupt, Andreas and Hartenstein, Justin and Reuel, Anka and Kochenderfer, Mykel J. and Koyejo, Sanmi},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT — see `LICENSE`.

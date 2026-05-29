.PHONY: all clean analysis figures tables refresh-judge robustness robustness-set1 robustness-set2

# Variables for refresh-judge — override on the command line:
#   make refresh-judge JUDGE=openai PERSPECTIVE=automation N_ITEMS=10
JUDGE ?= claude
PERSPECTIVE ?= both
N_ITEMS ?= 1

OUTDIR := outputs
ANALYSIS := python -m analysis

FIGURES := $(OUTDIR)/figure2_gwa_loadings.pdf \
           $(OUTDIR)/figure3_pareto_ecdf.pdf

TABLES := $(OUTDIR)/table1_cost_quantiles.tex \
          $(OUTDIR)/table2_noise_quantiles.tex \
          $(OUTDIR)/table4_noise_decomposition.tex

all: figures tables

figures: $(FIGURES)
tables: $(TABLES)

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OUTDIR)/figure2_gwa_loadings.pdf: analysis/figure2_gwa_quadrant.py data/judge_loadings.jsonl data/gwa_welfare_scores.csv | $(OUTDIR)
	$(ANALYSIS).figure2_gwa_quadrant

$(OUTDIR)/figure3_pareto_ecdf.pdf: analysis/figure3_pareto_ecdf.py data/welfare.csv | $(OUTDIR)
	$(ANALYSIS).figure3_pareto_ecdf

$(OUTDIR)/table1_cost_quantiles.tex: analysis/table1_cost_quantiles.py data/cost.csv | $(OUTDIR)
	$(ANALYSIS).table1_cost_quantiles

$(OUTDIR)/table2_noise_quantiles.tex: analysis/table2_noise_quantiles.py data/noise.csv | $(OUTDIR)
	$(ANALYSIS).table2_noise_quantiles

$(OUTDIR)/table4_noise_decomposition.tex: analysis/table4_noise_decomposition.py data/item_margins.parquet | $(OUTDIR)
	$(ANALYSIS).table4_noise_decomposition

# Re-run the LLM-as-judge GWA loadings. Costs real money; defaults to N_ITEMS=1
# as a smoke test. Requires OPENAI_API_KEY / ANTHROPIC_API_KEY (see env.example).
# Run `python -m analysis.fetch_olmes_items` first to materialize data/items.jsonl.
refresh-judge:
	python -m analysis.refresh_judge --judge $(JUDGE) --perspective $(PERSPECTIVE) --n $(N_ITEMS)

# Robustness checks. Both write CSVs to outputs/.
#   set1 — pipeline perturbations (top-k, weights, judge model, per-GWA
#          aggregation) on the headline 317-item set; reports rank
#          correlations, Pareto-frontier overlap, and the Finding-2
#          dominance statistic.
#   set2 — three independent raters (two human, one Claude with 3 examples)
#          on a 28-item held-out set; reports loading-mass per quadrant
#          (Finding 1) and math-vs-general welfare gap (Finding 2).
robustness: robustness-set1 robustness-set2

robustness-set1: analysis/robustness_set1.py data/judge_loadings.jsonl data/judge_loadings_openai.jsonl data/gwa_welfare_scores.csv data/gwa_welfare_scores_mean.csv | $(OUTDIR)
	$(ANALYSIS).robustness_set1

robustness-set2: analysis/robustness_set2.py data/heldout_raters/heldout_28_items.json data/heldout_raters/family_map.json data/gwa_welfare_scores.csv | $(OUTDIR)
	$(ANALYSIS).robustness_set2

clean:
	rm -rf $(OUTDIR)

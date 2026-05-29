"""
Table 1: Cost-ranking quantile bands at 4B scale, 7 items per quantile.

Reads:
  data/cost.csv         — items passing cost eligibility, with rank CIs.
  data/item_topics.csv  — short LLM topic summary per (task, doc_id).

Writes:
  outputs/table1_cost_quantiles.tex
  outputs/table1_cost_methodology.tex   — \\paragraph block, includable above the table.

Items without a cached topic summary appear with "(no summary)" — re-run the
upstream pipeline (pipeline/03_run_gwa_judge.py and pipeline/llm_summarize_items.py)
to refresh the cache after methodology changes that move new items into the bins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis._quantile_helpers import (
    QUANTILES, EXAMPLES_PER_BIN,
    pick_examples, humanize_cost, latex_escape, family_label,
    quantile_section_header,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


METHOD_TEX = r"""\paragraph{Cost ranking methodology.}
For each item $i$ that passes the noise + cost eligibility filters and has at
least one positive cost-axis slope, we compute
$\mathrm{cost}_i = \min_{k:\hat\beta_{i,k}>0} \kappa_k / \hat\beta_{i,k}$,
the marginal dollar cost to push item $i$'s probability of correct answer
$\hat p$ along the cheapest improving axis $k^\star \in \{\mathrm{PT},\mathrm{SFT},\mathrm{RL}\}$.
We report the cost to move $\hat p$ by $0.10$ (i.e.\ $\mathrm{cost}_i / 10$),
the local-linear regime of the OLS fit. Items are ranked ascending by
$\mathrm{cost}_i$. The bracketed range after each rank is the
$[2.5,\,97.5]\%$ percentile across $B=1000$ residual bootstrap replicates of
the per-item OLS, re-ranked within the headline set on each replicate.
"""


def render_cost_table(df: pd.DataFrame, picked: pd.DataFrame,
                      summaries: dict[tuple, str]) -> str:
    n_elig = len(df)
    caption = (
        f"Cost-ranking quantile bands at 4B scale: {EXAMPLES_PER_BIN} items per "
        f"quantile, ranked ascending by cost-per-10pp improvement. Bracketed "
        f"interval after each rank is the $[2.5,\\,97.5]\\%$ bootstrap rank CI. "
        f"Of {n_elig} items passing cost eligibility filters, items where every "
        f"axis has $\\hat\\beta_{{i,k}}\\le 0$ are excluded by formula constraint. "
        f"See methodology in adjacent \\texttt{{table1\\_cost\\_methodology.tex}}."
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        r"\label{tab:cost-quant-4B}",
        r"\begin{tabular}{rlp{0.42\linewidth}lr}",
        r"\toprule",
        r"\textbf{Rank [CI]} & \textbf{Benchmark} & \textbf{Topic} & "
        r"\textbf{Axis} & \textbf{Cost / 0.10 $\Delta\hat p$} \\",
        r"\midrule",
    ]
    last_q = None
    for _, r in picked.iterrows():
        if r["_quantile"] != last_q:
            if last_q is not None:
                lines.append(r"\midrule")
            lines.append(quantile_section_header(r["_quantile"], n_cols=5))
            lines.append(r"\midrule")
            last_q = r["_quantile"]
        topic = summaries.get((r["task"], int(r["doc_id"])), "(no summary)")
        rank_str = (f"{int(r['rank_headline'])}\\,"
                    f"[{int(r['rank_lo'])}--{int(r['rank_hi'])}]")
        cost_str = humanize_cost(r["cost_min_usd"] * 0.10)
        lines.append(
            f"  {rank_str} & "
            f"{latex_escape(family_label(r['task']))} & "
            f"{latex_escape(topic)} & "
            f"\\textsc{{{r['cheapest_axis']}}} & "
            f"{cost_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    cost = pd.read_csv(DATA / "cost.csv")
    cost = cost[cost["cost_min_usd"].notna() & (cost["cost_min_usd"] > 0)].reset_index(drop=True)

    topics = pd.read_csv(DATA / "item_topics.csv")
    summaries = {(r["task"], int(r["doc_id"])): r["topic"]
                 for _, r in topics.iterrows()}

    picked = pick_examples(cost, value_col="cost_min_usd")

    table_tex = render_cost_table(cost, picked, summaries)
    method_tex = METHOD_TEX

    (OUT / "table1_cost_quantiles.tex").write_text(table_tex)
    (OUT / "table1_cost_methodology.tex").write_text(method_tex)
    print(f"wrote {OUT / 'table1_cost_quantiles.tex'}")
    print(f"wrote {OUT / 'table1_cost_methodology.tex'}")


if __name__ == "__main__":
    main()

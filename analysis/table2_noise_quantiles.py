"""
Table 2: Noise-ranking quantile bands at 4B scale, 7 items per quantile.

Reads:
  data/noise.csv        — items passing noise eligibility, with rank CIs.
  data/item_topics.csv  — short LLM topic summary per (task, doc_id).

Writes:
  outputs/table2_noise_quantiles.tex
  outputs/table2_noise_methodology.tex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis._quantile_helpers import (
    QUANTILES, EXAMPLES_PER_BIN,
    pick_examples, humanize_sigma, latex_escape, family_label,
    quantile_section_header,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


METHOD_TEX = r"""\paragraph{Noise ranking methodology.}
For each item $i$ passing the noise eligibility filter we estimate
$\hat\sigma_i^2 = \frac{1}{N-K}\sum_t\sum_s
(\hat p_i^{(s,t)} - \bar p_i^{(t)})^2$, the pooled within-checkpoint
across-seed variance over $S=8$ PolyPythia~410M seeds and $K=10$ pretraining
checkpoints. Items are ranked ascending by $\hat\sigma_i$. The bracketed
range after each rank is the $[2.5,\,97.5]\%$ percentile of that item's rank
across $B=1000$ across-seed bootstrap replicates (resample seeds with
replacement within each checkpoint cluster, recompute $\hat\sigma_i^2$, then
re-rank).
"""


def render_noise_table(df: pd.DataFrame, picked: pd.DataFrame,
                       summaries: dict[tuple, str]) -> str:
    n_elig = len(df)
    caption = (
        f"Noise-ranking quantile bands at 4B scale: {EXAMPLES_PER_BIN} items per "
        f"quantile, ranked ascending by $\\hat\\sigma_i$. Bracketed interval after "
        f"each rank is the $[2.5,\\,97.5]\\%$ bootstrap rank CI. The table indexes "
        f"the {n_elig} items passing noise eligibility "
        f"($\\bar p \\in [0.10, 0.90]$ and $\\hat\\sigma^2 \\geq 10^{{-6}}$). "
        f"See methodology in adjacent \\texttt{{table2\\_noise\\_methodology.tex}}."
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        r"\label{tab:noise-quant-4B}",
        r"\begin{tabular}{rlp{0.42\linewidth}rr}",
        r"\toprule",
        r"\textbf{Rank [CI]} & \textbf{Benchmark} & \textbf{Topic} & "
        r"$\hat\sigma$ & $\bar{\hat p}$ \\",
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
        lines.append(
            f"  {rank_str} & "
            f"{latex_escape(family_label(r['task']))} & "
            f"{latex_escape(topic)} & "
            f"{humanize_sigma(r['noise_sigma_hat'])} & "
            f"{r['noise_mean_phat']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    noise = pd.read_csv(DATA / "noise.csv")
    topics = pd.read_csv(DATA / "item_topics.csv")
    summaries = {(r["task"], int(r["doc_id"])): r["topic"]
                 for _, r in topics.iterrows()}

    picked = pick_examples(noise, value_col="noise_sigma_hat")

    table_tex = render_noise_table(noise, picked, summaries)
    method_tex = METHOD_TEX

    (OUT / "table2_noise_quantiles.tex").write_text(table_tex)
    (OUT / "table2_noise_methodology.tex").write_text(method_tex)
    print(f"wrote {OUT / 'table2_noise_quantiles.tex'}")
    print(f"wrote {OUT / 'table2_noise_methodology.tex'}")


if __name__ == "__main__":
    main()

"""
Table 4: Per-family decomposition of seed-to-seed noise on the PolyPythias 410M
panel — Bernoulli component + adversarial-filtering / margin component.

Reads:
  data/item_margins.parquet — per-item gold/runner-up margin summary across
                              the 8 seeds × 10 checkpoints (precomputed).

Writes:
  outputs/table4_noise_decomposition.tex
  outputs/table4_noise_decomposition_writeup.tex

The parquet is precomputed by streaming over the raw PolyPythia 410M sample
JSONLs (~9.3 GB, not shipped). To regenerate from raw data, see
pipeline/06_estimate_noise_and_cost.py and pipeline/noise_decomposition_compute.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["HellaSwag", "PIQA", "ARC-C", "MathQA", "MMLU", "CSQA"]


def normalize_family(t: str) -> str:
    """Map raw task name to display family label."""
    if t == "hellaswag": return "HellaSwag"
    if t == "piqa": return "PIQA"
    if t == "arc_challenge": return "ARC-C"
    if t == "commonsense_qa": return "CSQA"
    if t == "mathqa": return "MathQA"
    if t.startswith("agieval_"): return "AGIEval"
    if t.startswith("mmlu_"): return "MMLU"
    return t


def family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in ORDER:
        sub = df[df["family"] == fam]
        if len(sub) == 0:
            continue
        sub_won = sub[sub["phat_noise"] > 0.5]
        # Within-family Pearson r between |ratio - 1| and Z_sigma
        if len(sub) >= 30:
            dist_from_tie = (sub["log_ratio_med"] - 1.0).abs()
            r_marg_z = float(np.corrcoef(dist_from_tie, sub["sigma_z"])[0, 1])
        else:
            r_marg_z = float("nan")
        rows.append({
            "Family": fam,
            "n": len(sub),
            "med_phat": sub["phat_noise"].median(),
            "med_sigma": sub["sigma"].median(),
            "med_sigma_z": sub["sigma_z"].median(),
            "n_won": len(sub_won),
            "med_log_ratio_won": (sub_won["log_ratio_med"].median()
                                  if len(sub_won) else float("nan")),
            "r_marg_z": r_marg_z,
        })
    return pd.DataFrame(rows)


def write_table(out: pd.DataFrame, path: Path) -> None:
    def f(v, prec=3):
        if pd.isna(v): return "--"
        return f"{v:.{prec}f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Decomposition of per-family seed-to-seed noise on the 410M PolyPythia panel "
        r"(8 seeds $\times$ 10 pretraining steps). \textbf{Bernoulli component:} median "
        r"$\hat p$ shows where each family's items sit on the probability scale; items far "
        r"from $\hat p=0.5$ have small $\hat\sigma$ mechanically. "
        r"$Z_{\hat\sigma}=\hat\sigma/\sqrt{\hat p(1-\hat p)}$ removes this dependence "
        r"($Z\approx 1/\sqrt{K}=0.32$ for pure binomial sampling at $K=10$). "
        r"\textbf{Margin component:} on items the model gets right ($\hat p>0.5$), "
        r"med.\ ratio = median of $\overline{\log p_{\text{gold}}}/\overline{\log p_{\text{runner-up}}}$ "
        r"in char-normalized log-prob (ratio $=1$ means a tie); $r$ is the within-family "
        r"Pearson correlation between $|\text{ratio}-1|$ and $Z_{\hat\sigma}$.}",
        r"\label{tab:noise-decomp}",
        r"\begin{tabular}{lrcc cccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Bernoulli component} & \multicolumn{4}{c}{Margin component} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
        r"Family & $n$ & med.\ $\hat p$ & med.\ $\hat\sigma$ "
        r"& med.\ $Z_{\hat\sigma}$ & $n_{\text{won}}$ & med.\ ratio & $r$ \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['Family']} & {int(r['n'])} & {f(r['med_phat'])} & {f(r['med_sigma'])} "
            f"& {f(r['med_sigma_z'])} & {int(r['n_won'])} "
            f"& {f(r['med_log_ratio_won'])} & {f(r['r_marg_z'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def write_writeup(path: Path) -> None:
    writeup = r"""\paragraph{What drives the per-family noise ranking?}
The raw seed-to-seed noise $\hat\sigma$ is dominated by HellaSwag, PIQA, and ARC-Challenge,
and is smallest on MMLU, CommonsenseQA, and MathQA. We decompose this ranking into two
components (Table~\ref{tab:noise-decomp}).

\textbf{(1) Bernoulli component.} For binary scores, $\sigma^2(\hat p)\approx \hat p(1-\hat p)/K$
peaks at $\hat p=0.5$. The 410M PolyPythia is near chance on MMLU/CSQA/MathQA (median
$\hat p\!\in\![0.15,0.24]$) but near $0.5$ on HellaSwag/PIQA (median $\hat p\!\in\![0.51,0.59]$),
which mechanically inflates $\hat\sigma$ on the latter. After Bernoulli normalization
$Z_{\hat\sigma}=\hat\sigma/\sqrt{\hat p(1-\hat p)}$, MMLU/CSQA/MathQA collapse to
$Z\approx 0.20$ -- below the binomial floor $1/\sqrt{K}\approx 0.32$ -- meaning their
seed-to-seed answers are essentially deterministic at this scale. HellaSwag remains
at $Z=0.62$, roughly twice the binomial floor.

\textbf{(2) Adversarial-filtering / margin component.} On items the model gets right
($\hat p>0.5$), we compute the ratio of gold to best-distractor char-normalized log-prob
(both quantities pass through the same normalization that defines \texttt{acc\_norm},
so the ratio is unit-fair). HellaSwag's median ratio is $0.997$, i.e.\ the gold beats the
best distractor by $0.3\%$ in nats/char. MMLU's is $0.669$, a $\sim$$100\times$ wider margin
in identical units. Within MMLU, $|$ratio$-1|$ is strongly anti-correlated with $Z_{\hat\sigma}$
($r=-0.57$): tighter margins reliably produce higher excess noise. Within HellaSwag this
correlation collapses to $r=-0.08$, consistent with adversarial filtering having driven
\emph{all} HellaSwag items to a tight-margin ceiling so margin no longer differentiates
noisy from quiet items.

The residual noise on HellaSwag/PIQA/ARC-C after controlling for $\hat p$ is therefore
attributable to benchmark construction, not to fundamental capability or item-format
artifacts: distractors are by design semantically close to gold in log-prob space, so
small seed-dependent perturbations in continuation log-likelihood routinely flip the
\texttt{acc\_norm} argmax.
"""
    path.write_text(writeup)
    print(f"wrote {path}")


def main():
    df = pd.read_parquet(DATA / "item_margins.parquet")
    df["family"] = df["task"].map(normalize_family)

    out = family_summary(df)
    print("\nPer-family decomposition:")
    print(out.round(3).to_string(index=False))

    write_table(out, OUT / "table4_noise_decomposition.tex")
    write_writeup(OUT / "table4_noise_decomposition_writeup.tex")


if __name__ == "__main__":
    main()

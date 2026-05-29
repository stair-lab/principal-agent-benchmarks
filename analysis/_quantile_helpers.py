"""Shared helpers for the cost / noise quantile tables (Tables 1 and 2)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
EXAMPLES_PER_BIN = 7


def humanize_cost(c: float) -> str:
    """No scientific notation. Returns LaTeX-ready string with escaped $."""
    if not np.isfinite(c) or c <= 0:
        return "—"
    if c < 1_000: return f"\\${c:.0f}"
    if c < 1_000_000:
        v = c / 1_000
        return f"\\${v:.1f}k" if v < 10 else f"\\${v:.0f}k"
    if c < 1_000_000_000:
        v = c / 1_000_000
        return f"\\${v:.1f}M" if v < 10 else f"\\${v:.0f}M"
    if c < 1_000_000_000_000:
        v = c / 1_000_000_000
        return f"\\${v:.1f}B" if v < 10 else f"\\${v:.0f}B"
    if c < 1e15:
        return f"\\${c/1e12:.0f}T"
    return f"\\${c:.0e}"


def humanize_sigma(s: float) -> str:
    if not np.isfinite(s):
        return "—"
    return f"{s:.3f}"


_LATEX_SPECIALS = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(s))


def family_label(task: str) -> str:
    """Short human-readable benchmark label (paper's column convention)."""
    if task == "mathqa": return "MathQA"
    if task == "hellaswag": return "HellaSwag"
    if task == "piqa": return "PIQA"
    if task == "arc_challenge": return "ARC-C"
    if task == "commonsense_qa": return "CSQA"
    if task.startswith("agieval_"): return "AGIEval-AQuA"
    if task.startswith("mmlu_"):
        return "MMLU/" + task.replace("mmlu_", "").replace("_", " ")
    return task


def pick_examples(df: pd.DataFrame, value_col: str,
                  quantiles: list[float] = QUANTILES,
                  k_per_bin: int = EXAMPLES_PER_BIN) -> pd.DataFrame:
    df_s = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    n = len(df_s)
    out_rows = []
    for q in quantiles:
        centre = int(round(q * (n - 1)))
        half = k_per_bin // 2
        lo = max(0, centre - half)
        hi = min(n, lo + k_per_bin)
        lo = max(0, hi - k_per_bin)
        for idx in range(lo, hi):
            r = df_s.iloc[idx].to_dict()
            r["_quantile"] = q
            out_rows.append(r)
    return pd.DataFrame(out_rows)


def quantile_section_header(q: float, n_cols: int) -> str:
    return (r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{quantile "
            f"{int(q*100)}\\%" + r"}} \\")

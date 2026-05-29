"""
Figure 2: GWA loadings on the WORKBank welfare landscape, automation- and
augmentation-framed (two-panel scatter).

Per-GWA axes are the bare WORKBank shares rescaled to [1, 5]
(p_auto_15, p_aug_15). Point size encodes the fraction of OLMES items whose
Claude top-5 GWA ranking includes that GWA.

Reads:
  data/judge_loadings.jsonl   — one row per (item, perspective): the LLM-judge's
                                top-5 ranked_gwa_ids.
  data/gwa_welfare_scores.csv — 24 retained GWAs with p_auto_15, p_aug_15.

Writes:
  outputs/figure2_gwa_loadings.{png,pdf}
  outputs/figure2_gwa_loadings.csv
  outputs/figure2_gwa_loadings_findings.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


THRESHOLD = 3.5

QUADRANTS = {
    "green":    ("hi_auto", "hi_aug", "Green-Light"),
    "automate": ("hi_auto", "lo_aug", "Automate-Only"),
    "augment":  ("lo_auto", "hi_aug", "Augment-Only"),
    "low":      ("lo_auto", "lo_aug", "Red-Light"),
}
QUAD_COLORS = {
    "green":    "#2ca02c",
    "automate": "#ff7f0e",
    "augment":  "#1f77b4",
    "low":      "#d62728",
}


def quadrant_of(p_auto: float, p_aug: float) -> str:
    if p_auto >= THRESHOLD and p_aug >= THRESHOLD: return "green"
    if p_auto >= THRESHOLD and p_aug <  THRESHOLD: return "automate"
    if p_auto <  THRESHOLD and p_aug >= THRESHOLD: return "augment"
    return "low"


def compute_loading_fractions(rows: list[dict], kept_gwas: list[str],
                              perspective: str, top_k: int = 5
                              ) -> tuple[dict[str, float], int]:
    """Fraction of items whose top-k ranking contains each GWA, in the given
    perspective. Returns (frac_dict, n_items)."""
    item_loaded: dict[str, set[tuple]] = defaultdict(set)
    items_seen: set[tuple] = set()
    for r in rows:
        if not r.get("ok") or r.get("perspective") != perspective:
            continue
        key = (r["lm_eval_task"], r["doc_id"])
        items_seen.add(key)
        for gid in (r.get("ranked_gwa_ids") or [])[:top_k]:
            if gid:
                item_loaded[gid].add(key)
    n = len(items_seen)
    return {g: len(item_loaded.get(g, set())) / max(1, n) for g in kept_gwas}, n


GWA_SHORT = {
    "performing_administrative":              "Admin work",
    "documenting_recording":                  "Documenting",
    "judging_qualities":                      "Judging quality",
    "monitoring_processes":                   "Monitoring proc.",
    "guiding_directing_motivating":           "Guiding/Directing",
    "scheduling_work":                        "Scheduling",
    "getting_information":                    "Getting info",
    "assisting_caring":                       "Assisting",
    "processing_information":                 "Processing info",
    "communicating_internally":               "Comm. internally",
    "updating_using_knowledge":               "Updating knowledge",
    "analyzing_data":                         "Analyzing data",
    "thinking_creatively":                    "Thinking creatively",
    "developing_objectives_strategies":       "Dev. objectives",
    "evaluating_compliance":                  "Eval. compliance",
    "estimating_quantifiable_characteristics":"Est. quant.",
    "providing_consultation":                 "Consultation",
    "making_decisions_solving_problems":      "Making decisions",
    "communicating_externally":               "Comm. externally",
    "monitoring_controlling_resources":       "Mon. resources",
    "interpreting_information":               "Interpret. info",
    "performing_for_public":                  "Public-facing",
    "selling_influencing":                    "Selling",
    "staffing_organizational_units":          "Staffing",
}


# Label-position overrides under the no-breadth axes (p_auto_15, p_aug_15).
# Override format: (dx, dy, ha, va) — anchor offset in data coords + alignment.
LABEL_OVERRIDES: dict[tuple[str, str], tuple[float, float, str, str]] = {
    # Bottom-left collisions: Public-facing + Staffing.
    ("automation",   "Public-facing"):       ( 0.15,  0.00, "left",   "center"),
    ("augmentation", "Public-facing"):       ( 0.15,  0.00, "left",   "center"),
    ("automation",   "Staffing"):            ( 0.05,  0.20, "left",   "bottom"),
    ("augmentation", "Staffing"):            ( 0.05,  0.20, "left",   "bottom"),
    # Updating knowledge: redirect upward (default would clip the right edge).
    ("automation",   "Updating knowledge"):  (-0.05,  0.20, "right",  "bottom"),
    ("augmentation", "Updating knowledge"):  (-0.05,  0.20, "right",  "bottom"),
    # Bottom-right corner: Scheduling and Assisting both at y≈1.
    ("automation",   "Scheduling"):          (-0.05,  0.18, "right",  "bottom"),
    ("augmentation", "Scheduling"):          (-0.05,  0.18, "right",  "bottom"),
    ("automation",   "Assisting"):           ( 0.05,  0.18, "left",   "bottom"),
    ("augmentation", "Assisting"):           ( 0.05,  0.18, "left",   "bottom"),
    # Guiding/Directing: below its dot to clear top edge.
    ("automation",   "Guiding/Directing"):   (-0.05, -0.20, "right",  "top"),
    ("augmentation", "Guiding/Directing"):   (-0.05, -0.20, "right",  "top"),
    # Dev. objectives: just above its dot.
    ("automation",   "Dev. objectives"):     ( 0.05,  0.05, "left",   "bottom"),
    ("augmentation", "Dev. objectives"):     ( 0.05,  0.05, "left",   "bottom"),
    # Thinking creatively: just below its dot.
    ("automation",   "Thinking creatively"): ( 0.05, -0.10, "left",   "top"),
    ("augmentation", "Thinking creatively"): ( 0.05, -0.10, "left",   "top"),
    # Getting info: above-right of dot.
    ("automation",   "Getting info"):        ( 0.05,  0.10, "left",   "bottom"),
    ("augmentation", "Getting info"):        ( 0.05,  0.10, "left",   "bottom"),
    # Comm. internally: push into the empty Green-Light area.
    ("automation",   "Comm. internally"):    ( 0.10,  0.18, "left",   "bottom"),
    ("augmentation", "Comm. internally"):    ( 0.10,  0.18, "left",   "bottom"),
    # Eval. compliance: bring closer to dot.
    ("automation",   "Eval. compliance"):    (-0.04, -0.08, "right",  "top"),
    ("augmentation", "Eval. compliance"):    (-0.04, -0.08, "right",  "top"),
}


def _place_labels(ax, points: list[tuple[float, float, str]],
                  fontsize: float = 16.0,
                  perspective: str | None = None) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_size_data(text: str) -> tuple[float, float]:
        t = ax.text(0, 0, text, fontsize=fontsize, alpha=0)
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        x0, y0 = inv.transform((bb.x0, bb.y0))
        x1, y1 = inv.transform((bb.x1, bb.y1))
        return abs(x1 - x0), abs(y1 - y0)

    pad_radial = 0.12
    base_offsets = [
        ( 1.0,  0.0), ( 0.85,  0.85), ( 0.0,  1.0), (-0.85,  0.85),
        (-1.0,  0.0), (-0.85, -0.85), ( 0.0, -1.0), ( 0.85, -0.85),
    ]
    point_xy = np.array([(x, y) for (x, y, _) in points])
    placed_bboxes: list[tuple[float, float, float, float]] = []

    for x, y, text in points:
        w, h = text_size_data(text)

        override = LABEL_OVERRIDES.get((perspective, text)) if perspective else None
        if override is not None:
            dx_data, dy_data, ha, va = override
            anchor_x, anchor_y = x + dx_data, y + dy_data
            if ha == "left":     bb_x0, bb_x1 = anchor_x, anchor_x + w
            elif ha == "right":  bb_x0, bb_x1 = anchor_x - w, anchor_x
            else:                bb_x0, bb_x1 = anchor_x - w/2, anchor_x + w/2
            if va == "bottom":   bb_y0, bb_y1 = anchor_y, anchor_y + h
            elif va == "top":    bb_y0, bb_y1 = anchor_y - h, anchor_y
            else:                bb_y0, bb_y1 = anchor_y - h/2, anchor_y + h/2
            placed_bboxes.append((bb_x0, bb_y0, bb_x1, bb_y1))
            ax.annotate(text, (x, y), xytext=(anchor_x, anchor_y), textcoords="data",
                        fontsize=fontsize, color="#1a1a1a", ha=ha, va=va, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.85))
            continue

        best_score = float("inf")
        best_anchor = (x + pad_radial, y + pad_radial)
        best_dir = (1, 1)
        best_bbox = (x, y, x + w, y + h)

        for dx, dy in base_offsets:
            ax_pad_x = pad_radial * 1.0
            ax_pad_y = pad_radial * 0.6
            ax_x = x + dx * ax_pad_x
            ax_y = y + dy * ax_pad_y

            if dx > 0:   x0, x1 = ax_x, ax_x + w
            elif dx < 0: x0, x1 = ax_x - w, ax_x
            else:        x0, x1 = ax_x - w/2, ax_x + w/2
            if dy > 0:   y0, y1 = ax_y, ax_y + h
            elif dy < 0: y0, y1 = ax_y - h, ax_y
            else:        y0, y1 = ax_y - h/2, ax_y + h/2

            score = 0.0
            if x0 < 1.0: score += (1.0 - x0) * 50
            if x1 > 5.0: score += (x1 - 5.0) * 50
            if y0 < 1.0: score += (1.0 - y0) * 50
            if y1 > 5.0: score += (y1 - 5.0) * 50
            if x0 < THRESHOLD < x1:
                score += min(x1 - THRESHOLD, THRESHOLD - x0) * 6
            if y0 < THRESHOLD < y1:
                score += min(y1 - THRESHOLD, THRESHOLD - y0) * 6
            for (qx, qy) in point_xy:
                if x0 < qx < x1 and y0 < qy < y1:
                    score += 4
            for (bx0, by0, bx1, by1) in placed_bboxes:
                ox = max(0, min(x1, bx1) - max(x0, bx0))
                oy = max(0, min(y1, by1) - max(y0, by0))
                if ox > 0 and oy > 0:
                    score += (ox * oy) * 200
            score += 0.05 * (abs(dx) + abs(dy))

            if score < best_score:
                best_score = score
                best_anchor = (ax_x, ax_y)
                best_dir = (dx, dy)
                best_bbox = (x0, y0, x1, y1)

        placed_bboxes.append(best_bbox)
        dx, dy = best_dir
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
        ax.annotate(text, (x, y), xytext=best_anchor, textcoords="data",
                    fontsize=fontsize, color="#1a1a1a", ha=ha, va=va, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))


def plot_quadrant(gwa: pd.DataFrame, frac_auto: dict[str, float],
                  frac_aug: dict[str, float], out_path: Path,
                  x_col: str = "p_auto_15", y_col: str = "p_aug_15") -> None:
    fig = plt.figure(figsize=(24, 13.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=(1.0, 0.06), hspace=0.16, wspace=0.06,
        left=0.045, right=0.99, top=0.96, bottom=0.10,
    )
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_l)
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")

    cs_auto = np.array([frac_auto.get(g, 0) for g in gwa["concept_id"]])
    cs_aug  = np.array([frac_aug.get(g, 0)  for g in gwa["concept_id"]])
    vmax = float(max(0.05, cs_auto.max(), cs_aug.max()))

    for ax, cs, persp in zip(
        [ax_l, ax_r], [cs_auto, cs_aug], ["automation", "augmentation"]
    ):
        for q, (xlo, xhi, ylo, yhi) in [
            ("low",      (1, THRESHOLD, 1, THRESHOLD)),
            ("automate", (THRESHOLD, 5, 1, THRESHOLD)),
            ("augment",  (1, THRESHOLD, THRESHOLD, 5)),
            ("green",    (THRESHOLD, 5, THRESHOLD, 5)),
        ]:
            ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                   facecolor=QUAD_COLORS[q], alpha=0.10,
                                   zorder=0, linewidth=0))
        ax.axvline(THRESHOLD, color="#444", lw=1, ls="--", zorder=1)
        ax.axhline(THRESHOLD, color="#444", lw=1, ls="--", zorder=1)

        for q, (cx, cy, ha, va) in [
            ("low",      (1.05, 1.05, "left",  "bottom")),
            ("automate", (4.95, 1.05, "right", "bottom")),
            ("augment",  (1.05, 4.95, "left",  "top")),
            ("green",    (4.95, 4.95, "right", "top")),
        ]:
            ax.text(cx, cy, QUADRANTS[q][2],
                    color=QUAD_COLORS[q],
                    fontsize=18, ha=ha, va=va, fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="white",
                              edgecolor=QUAD_COLORS[q],
                              alpha=0.9, lw=1.0))

        xs = gwa[x_col].values
        ys = gwa[y_col].values
        sizes = 70 + cs * 700
        ax.scatter(xs, ys, s=sizes,
                   facecolor="#5a5a5a", edgecolor="black", linewidth=0.7,
                   alpha=0.85, zorder=3)

        labels = [(row[x_col], row[y_col],
                   GWA_SHORT.get(row["concept_id"], row["concept_id"]))
                  for _, row in gwa.iterrows()]
        idx = np.argsort(-cs)
        labels = [labels[i] for i in idx]
        _place_labels(ax, labels, perspective=persp)

        ax.set_xlabel(r"$p_{\mathrm{auto}}$ (per-GWA automation desire)",
                      fontsize=24)
        if persp == "automation":
            ax.set_ylabel(r"$p_{\mathrm{aug}}$ (per-GWA augmentation desire)",
                          fontsize=24)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlim(0.9, 5.1); ax.set_ylim(0.9, 5.1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        if persp == "augmentation":
            plt.setp(ax.get_yticklabels(), visible=False)

    ref_fracs = [0.05, 0.25, 0.50, 1.00]
    ref_fracs = [f for f in ref_fracs if f <= max(vmax, 1.0)]
    legend_handles = [
        plt.scatter([], [], s=70 + f * 700,
                    facecolor="#5a5a5a", edgecolor="black", linewidth=0.7,
                    alpha=0.85,
                    label=f"{int(f*100)}%")
        for f in ref_fracs
    ]
    leg = leg_ax.legend(
        handles=legend_handles,
        title="fraction of OLMES items whose top-5 GWA ranking includes this GWA",
        loc="center", ncol=len(ref_fracs), frameon=False,
        fontsize=18, title_fontsize=22, labelspacing=1.0,
        handletextpad=1.2, columnspacing=2.5, borderpad=0.5,
    )
    leg._legend_box.align = "center"

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    print(f"[plot] wrote {pdf_path}")


def quadrant_breakdown(gwa: pd.DataFrame, frac: dict[str, float], label: str,
                       x_col: str, y_col: str) -> str:
    g = gwa.copy()
    g["quadrant"] = g.apply(lambda r: quadrant_of(r[x_col], r[y_col]), axis=1)
    g["frac"] = g["concept_id"].map(frac)
    L = [f"=== {label} ==="]
    for q in ["green", "automate", "augment", "low"]:
        sub = g[g["quadrant"] == q]
        n_gwa = len(sub)
        mean_frac = sub["frac"].mean() if n_gwa else 0.0
        max_frac = sub["frac"].max() if n_gwa else 0.0
        loaded = sub[sub["frac"] >= 0.10]["concept_id"].tolist()
        L.append(f"  {q:8s}  n_GWAs={n_gwa:2d}   mean loading={mean_frac:.3f}   "
                 f"max loading={max_frac:.3f}   GWAs ≥10% loaded: {len(loaded)}")
        for gid in loaded:
            row = sub[sub["concept_id"] == gid].iloc[0]
            L.append(f"      {gid:42s}  loading={row['frac']:.2f}  "
                     f"({x_col}={row[x_col]:.2f}, {y_col}={row[y_col]:.2f})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / "judge_loadings.jsonl").open()]

    gwa = pd.read_csv(DATA / "gwa_welfare_scores.csv")
    gwa = gwa[~gwa["dropped"]].reset_index(drop=True)
    kept_gwas = gwa["concept_id"].tolist()

    frac_auto, n_auto = compute_loading_fractions(rows, kept_gwas, "automation", args.top_k)
    frac_aug,  n_aug  = compute_loading_fractions(rows, kept_gwas, "augmentation", args.top_k)
    print(f"[load] {n_auto} items with automation calls, {n_aug} with augmentation calls")

    out_png = OUT / "figure2_gwa_loadings.png"
    plot_quadrant(gwa, frac_auto, frac_aug, out_png)

    x_col, y_col = "p_auto_15", "p_aug_15"
    gwa_out = gwa[["concept_id", "gwa", x_col, y_col]].copy()
    gwa_out["quadrant"] = gwa_out.apply(lambda r: quadrant_of(r[x_col], r[y_col]), axis=1)
    gwa_out["frac_loaded_auto_persp"] = gwa_out["concept_id"].map(frac_auto)
    gwa_out["frac_loaded_aug_persp"]  = gwa_out["concept_id"].map(frac_aug)
    gwa_out = gwa_out.sort_values(["quadrant", "frac_loaded_auto_persp"],
                                  ascending=[True, False])
    csv_path = OUT / "figure2_gwa_loadings.csv"
    gwa_out.to_csv(csv_path, index=False)
    print(f"[csv]  wrote {csv_path}")

    findings = [
        f"Headline coverage statistics (judge: claude, no-breadth axes)",
        "-" * 60,
        quadrant_breakdown(gwa, frac_auto, "automation-framed judge ranking", x_col, y_col),
        "",
        quadrant_breakdown(gwa, frac_aug, "augmentation-framed judge ranking", x_col, y_col),
    ]
    findings_path = OUT / "figure2_gwa_loadings_findings.txt"
    findings_path.write_text("\n".join(findings) + "\n")
    print(f"[txt]  wrote {findings_path}")


if __name__ == "__main__":
    main()

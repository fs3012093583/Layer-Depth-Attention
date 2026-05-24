"""
Generate Figure 3e: Joint Attention Matrix Comparison (Baseline vs Ours).

Reads attention matrix summary JSONs from two analysis output directories
and produces a side-by-side comparison figure for the paper.

Usage:
    python scripts/plot_figure3e_joint_attention_comparison.py

Output:
    paper/figures/figure3e_joint_attention_baseline_vs_ours_v6.pdf
    paper/figures/figure3e_joint_attention_baseline_vs_ours_v6.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_JSON = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "baseline_16l_step40000_matrix_l13_s64"
    / "attention_matrix_summary.json"
)
OURS_JSON = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "shared_kv_depth_memory_dualq_sublayer_16l_step40000_matrix_l13_s64"
    / "attention_matrix_summary.json"
)
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUT_STEM = "figure3e_joint_attention_baseline_vs_ours_v8"

# ---------------------------------------------------------------------------
# Global style (publication-ready)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

CMAP = "viridis"


def load_matrices(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    row = np.array(payload["row_matrix"], dtype=float)
    depth_raw = payload.get("depth_matrix")
    depth = None if depth_raw is None else np.array(depth_raw, dtype=float)
    joint = np.array(payload["joint_matrix"], dtype=float)
    return {"row": row, "depth": depth, "joint": joint, "payload": payload}


def nice_ticks(n: int, target_step: int = 16) -> np.ndarray:
    """Return tick positions at multiples of *target_step* within [0, n-1].
    Never includes n-1 unless it happens to be a multiple of target_step.
    Always starts at 0.
    """
    ticks = list(range(0, n, target_step))
    return np.array(ticks)


def plot_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    vmin: float,
    vmax: float,
    split_col: int | None = None,
) -> mpl.image.AxesImage:
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    seq_rows, total_cols = mat.shape

    # y-axis (query tokens) – step-16 ticks only, never include n-1 tail
    yticks = nice_ticks(seq_rows)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks], fontweight="bold")

    # x-axis
    if split_col is not None:
        # Left part: row tokens; right part: depth slots
        n_right = total_cols - split_col
        left_ticks = nice_ticks(split_col) if split_col > 0 else np.array([], int)
        # depth slots are only 24 wide, use step 8
        right_local = nice_ticks(n_right, target_step=8) if n_right > 0 else np.array([], int)
        right_ticks = split_col + right_local

        xticks = np.concatenate([left_ticks, right_ticks])
        xlabels = [str(t) for t in left_ticks] + [str(t) for t in right_local]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontweight="bold")

        # Dividing line between row-tokens and depth-memory regions
        ax.axvline(split_col - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.7)
    else:
        xticks = nice_ticks(total_cols)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontweight="bold")

    ax.tick_params(direction="out", length=3)
    return im


def main() -> None:
    baseline = load_matrices(BASELINE_JSON)
    ours = load_matrices(OURS_JSON)

    b_mat = baseline["joint"]  # (64, 64) – baseline has no depth slots
    o_mat = ours["joint"]      # (64, 88) – 64 row + 24 depth
    o_split = ours["row"].shape[1]  # == 64

    # Fixed color scale [0, 0.2] for reproducibility across figures.
    vmin = 0.0
    vmax = 0.2

    # ------------------------------------------------------------------
    # Layout: 1 row × 3 axes (baseline | ours | colorbar)
    # The "ours" axis is wider in proportion to its extra depth columns.
    # ------------------------------------------------------------------
    b_cols = b_mat.shape[1]          # 64
    o_cols = o_mat.shape[1]          # 88
    ratio = o_cols / b_cols          # ~1.375

    # Figure dimensions: keep the baseline panel ~3.5 in wide
    panel_h = 3.0   # inches – height of each heatmap
    b_w = 3.5       # inches – width of baseline panel
    o_w = b_w * ratio  # width of ours panel
    cbar_w = 0.45   # colorbar width
    gap_mid = 0.65  # gap between panels (for shared y-label space)
    margin_l = 0.62
    margin_r = 0.12

    # Extra vertical space (kept small – one text line each):
    #   top_title  – row-1 header ("Baseline" / "Ours")         ~14 pt
    #   top_sub    – row-2 sub-headers ("Row Tokens" etc.)       ~11 pt
    #   bot        – x-axis label + tick labels
    top_title = 0.20   # reduced: just enough for one 14-pt bold line
    top_sub   = 0.18   # reduced: just enough for one 11-pt bold line
    top_pad   = 0.04   # tiny breathing room above panel top
    bot       = 0.52

    fig_w = margin_l + b_w + gap_mid + o_w + cbar_w + margin_r
    fig_h = top_title + top_sub + top_pad + panel_h + bot

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Convert to [0,1] figure coordinates
    def x(px):
        return px / fig_w

    def y(py):
        return py / fig_h

    ax_b = fig.add_axes(
        [x(margin_l), y(bot), x(b_w), y(panel_h)]
    )
    ax_o = fig.add_axes(
        [x(margin_l + b_w + gap_mid), y(bot), x(o_w), y(panel_h)]
    )
    ax_cb = fig.add_axes(
        [x(margin_l + b_w + gap_mid + o_w + 0.07), y(bot), x(0.13), y(panel_h)]
    )

    # Panel top edge in figure coords (used to anchor text rows)
    pan_top = bot + panel_h

    im_b = plot_heatmap(ax_b, b_mat, vmin, vmax)
    im_o = plot_heatmap(ax_o, o_mat, vmin, vmax, split_col=o_split)

    ax_b.set_ylabel("Query Token Index", fontsize=12, fontweight="bold", labelpad=4)
    ax_b.set_xlabel("Attention Key Index", fontsize=12, fontweight="bold", labelpad=4)
    ax_o.set_xlabel("Attention Key Index", fontsize=12, fontweight="bold", labelpad=4)
    ax_o.set_yticklabels([])  # share y-scale, suppress duplicate labels

    # Colorbar
    cbar = fig.colorbar(im_b, cax=ax_cb)
    cbar.set_label("Attention Weight", fontsize=10, fontweight="bold", labelpad=5)
    cbar.ax.tick_params(labelsize=9)

    # ------------------------------------------------------------------
    # Row-1 titles: "Baseline" / "Ours"  (placed in figure coords)
    # ------------------------------------------------------------------
    b_cx = x(margin_l + b_w / 2)
    o_cx = x(margin_l + b_w + gap_mid + o_w / 2)

    # Title row sits in the top_title band above (top_sub + top_pad)
    title_y = y(pan_top + top_pad + top_sub + top_title * 0.52)

    fig.text(
        b_cx, title_y,
        "Baseline",
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        fontfamily="Times New Roman",
    )
    fig.text(
        o_cx, title_y,
        "Ours",
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        fontfamily="Times New Roman",
    )

    # ------------------------------------------------------------------
    # Row-2 sub-headers: "Row Tokens" / "Depth Slots"
    # ------------------------------------------------------------------
    # Sub-header row sits in the top_sub band, just above panel top
    sub_y = y(pan_top + top_pad + top_sub * 0.52)

    # Baseline: single sub-header centred over panel
    fig.text(
        b_cx, sub_y,
        "Row Tokens",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        fontfamily="Times New Roman",
    )

    # Ours: "Row Tokens" over left portion, "Depth Slots" over right portion
    o_row_cx = x(margin_l + b_w + gap_mid + o_split / o_cols * o_w / 2)
    o_dep_cx = x(margin_l + b_w + gap_mid + (o_split + (o_cols - o_split) / 2) / o_cols * o_w)

    fig.text(
        o_row_cx, sub_y,
        "Row Tokens",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        fontfamily="Times New Roman",
    )
    fig.text(
        o_dep_cx, sub_y,
        "Depth Slots",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        fontfamily="Times New Roman",
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"{OUT_STEM}.{ext}"
        dpi = 600 if ext == "png" else None
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )
        print(f"saved → {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

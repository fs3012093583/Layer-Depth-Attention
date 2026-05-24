from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def arrow(ax, start, end, color, lw=1.2, alpha=1.0, ls="-", ms=9, zorder=1, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        shrinkA=5,
        shrinkB=5,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)


def right_side_curve(ax, start_y, end_y, color, lw=1.5, alpha=0.9, ls="--", ms=9, zorder=2):
    """Draw same-position layer retrieval around the right side of the node column."""
    x0 = 4.16
    x_mid = 4.78
    verts = [
        (x0, start_y),
        (x_mid, start_y + 0.35),
        (x_mid, end_y - 0.35),
        (x0, end_y),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    patch = FancyArrowPatch(
        path=MplPath(verts, codes),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=2,
        shrinkB=5,
    )
    ax.add_patch(patch)


def draw_grid(ax, ours=False):
    n_tok = 5
    n_layer = 4
    xs = list(range(n_tok))
    ys = list(range(n_layer))

    node_edge = "#1f2933"
    row = "#2b6cb0"
    depth = "#c2410c"
    path = "#15803d"
    target = (4, 3)
    source = (0, 0)
    bridge = (4, 1)

    # Causal token-axis attention into the target at the top layer.
    for x in xs[:-1]:
        arrow(ax, (x, 2), target, row, lw=1.05, alpha=0.5, ms=7)

    # Local top-layer self path to the target.
    arrow(ax, (4, 2), target, row, lw=1.05, alpha=0.5, ms=7)

    if ours:
        # Same-position layer-depth retrieval edges into the target.
        right_side_curve(ax, 0.0, 3.0, depth, lw=1.45, alpha=0.78, ls="-", ms=8, zorder=2)
        right_side_curve(ax, 1.0, 3.0, depth, lw=1.45, alpha=0.78, ls="-", ms=8, zorder=2)

    # Highlight one example path from the lower-left source to the upper-right target.
    # Curvature keeps the path visually separate from the background routing edges.
    arrow(ax, source, bridge, path, lw=2.0, alpha=0.95, ls="--", ms=10, zorder=4, rad=0.18)
    if ours:
        arrow(ax, bridge, target, path, lw=2.0, alpha=0.95, ls="--", ms=10, zorder=4, rad=-0.42)
    else:
        arrow(ax, bridge, (4, 2), path, lw=2.0, alpha=0.95, ls="--", ms=10, zorder=4, rad=-0.36)
        arrow(ax, (4, 2), target, path, lw=2.0, alpha=0.95, ls="--", ms=10, zorder=4, rad=-0.36)

    # Draw nodes last.
    for x in xs:
        for y in ys:
            face = "#ffffff"
            edge = "#aab2bd"
            alpha = 0.42
            size = 32
            lw = 0.75
            if (x, y) == target:
                face = "#fef3c7"
                edge = node_edge
                alpha = 1.0
                size = 70
                lw = 1.25
            elif y == 2:
                face = "#eff6ff"
                edge = row
                alpha = 0.82
                size = 44
                lw = 0.95
            elif (x, y) in {source, bridge, (4, 2)}:
                face = "#ecfdf5"
                edge = path
                alpha = 1.0
                size = 54
                lw = 1.1
            ax.scatter(
                x,
                y,
                s=size,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=3,
            )

    ax.text(
        target[0] + 0.13,
        target[1] + 0.08,
        r"$x_t^{(L)}$",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        source[0] - 0.05,
        source[1] - 0.16,
        r"$x_1^{(0)}$",
        fontsize=7.5,
        fontweight="bold",
        ha="right",
        va="top",
        color=path,
    )
    # Coordinate axes clarify the token-position by layer-depth grid.
    ax.annotate(
        "",
        xy=(4.65, -0.42),
        xytext=(-0.22, -0.42),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111827", mutation_scale=8),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.45, 3.3),
        xytext=(-0.45, -0.22),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111827", mutation_scale=8),
        annotation_clip=False,
    )
    ax.text(2.2, -0.62, "Token position", fontsize=8.5, fontweight="bold", ha="center", va="top")
    ax.text(-0.62, 1.55, "Layer", fontsize=8.5, fontweight="bold", ha="center", va="center", rotation=90)

    ax.set_xlim(-0.8, 4.95)
    ax.set_ylim(-0.82, 3.78)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact legend-like labels.
    legend_y = 3.58
    ax.plot([0.05, 0.38], [legend_y, legend_y], color=path, lw=1.8, ls="--", clip_on=False)
    ax.text(0.46, legend_y, "shortest path", color=path, fontsize=7.3, fontweight="bold", va="center")
    ax.plot([0.05, 0.38], [legend_y - 0.23, legend_y - 0.23], color=row, lw=1.4, clip_on=False)
    ax.text(0.46, legend_y - 0.23, "token-axis attention", color=row, fontsize=7.3, fontweight="bold", va="center")
    if ours:
        ax.plot([0.05, 0.38], [legend_y - 0.46, legend_y - 0.46], color=depth, lw=1.5, ls="-", clip_on=False)
        ax.text(0.46, legend_y - 0.46, "layer-axis attention", color=depth, fontsize=7.3, fontweight="bold", va="center")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    draw_grid(axes[0], ours=False)
    draw_grid(axes[1], ours=True)

    axes[0].set_title(
        "(a) Standard Transformer",
        fontsize=10,
        fontweight="bold",
        pad=5,
    )
    axes[1].set_title(
        "(b) Layer-Depth Routing",
        fontsize=10,
        fontweight="bold",
        pad=5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), w_pad=1.4)

    png_path = OUT_DIR / "figure1_information_propagation_schematic_v8.png"
    pdf_path = OUT_DIR / "figure1_information_propagation_schematic_v8.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()

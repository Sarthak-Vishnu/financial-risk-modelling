# CAPTION: Every encoder configuration measured as a difference from the defended model,
# \texttt{struct+tfidf [sparse]}, on the 2025 validation year, grouped by what the encoder
# was trained to do. The shaded band is the defended model's own bootstrap confidence
# interval, half-width $0.068$. A filled marker is significantly less accurate than the
# defended model under a Diebold--Mariano test; a hollow marker cannot be separated from
# it. Every configuration falls inside the band, so no difference in the grid is
# resolvable on one year of data.

import matplotlib.pyplot as plt

from _grid import REFERENCE, by_family, load_grid, noise_halfwidth
from _style import C, TEXTWIDTH_IN, apply_style, save

ALPHA = 0.05

def label_for(name):
    return name[len("struct+"):] if name.startswith("struct+") else name

def main():
    apply_style()

    grid = load_grid()
    ref_ic = grid[REFERENCE]["ic"]
    band = noise_halfwidth(grid)
    families = by_family(grid)

    # lay out rows bottom-up, leaving a slot above each family for its header
    rows, headers, y = [], [], 0.0
    for title, members in reversed(families):
        for r in reversed(members):
            rows.append((y, label_for(r["name"]), r["ic"] - ref_ic, r["dm_p"]))
            y += 1.0
        headers.append((y - 0.35, title))
        y += 1.1

    # Height tracks the row and header count so the pitch stays constant as the grid
    # grows. 0.245in per slot plus fixed furniture reproduces the tuned pitch.
    height = 2.1 + 0.245 * y
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, height))

    ax.axvspan(-band, band, color=C["faint"], alpha=0.28, linewidth=0, zorder=0)
    ax.axvline(0.0, color=C["blue"], linestyle="--", linewidth=1.8, zorder=1)

    for ypos, label, delta, dm in rows:
        worse = dm is not None and dm < ALPHA
        colour = C["vermillion"] if worse else C["blue"]
        ax.plot([0.0, delta], [ypos, ypos], color=colour, linewidth=1.6, alpha=0.55,
                zorder=2)
        ax.scatter([delta], [ypos], s=95, zorder=3, color=colour,
                   facecolors=colour if worse else "white",
                   edgecolors=colour, linewidths=1.9)
        ax.text(delta - 0.0035 if delta < 0 else delta + 0.0035, ypos,
                f"{delta:+.3f}", fontsize=9.5, color=colour, va="center",
                ha="right" if delta < 0 else "left")

    for ypos, title in headers:
        ax.text(-band - 0.012, ypos, title, fontsize=11.5, style="italic",
                color="#333333", ha="left", va="center")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontfamily="monospace", fontsize=9.5)
    ax.set_ylim(-0.9, y - 0.5)
    ax.set_xlim(-band - 0.014, band + 0.014)
    ax.set_xlabel(r"IC difference from struct+tfidf [sparse]")
    ax.grid(axis="y", visible=False)

    xt = ax.get_xaxis_transform()
    ax.text(0.0, 1.015, REFERENCE, transform=xt, color=C["blue"],
            fontsize=10.5, ha="center", va="bottom", fontfamily="monospace")
    # Named at the band's LEFT edge, along the bottom. The right edge is no longer free:
    # the bottom row's difference is positive, so its value label sits bottom-right, and
    # the top-right corner is under the first family header.
    ax.text(-band + 0.002, 0.012, "single-year noise band", transform=xt,
            color="#555555", fontsize=10.5, ha="left", va="bottom")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=C["vermillion"],
                   label=rf"significantly worse, DM $p < {ALPHA:g}$"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor="white", markeredgecolor=C["blue"],
                   markeredgewidth=1.9, color=C["blue"],
                   label="tie: not separable from the reference"),
    ]
    # bbox_to_anchor is in axes fractions, so a fixed offset would drift in inches as the
    # figure grows. Scale it to hold the gap at what -0.30 gave on the 6.0in canvas.
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.30 * 6.0 / height), ncol=1)

    save(fig, "fig2_extra_delta_by_family", exact_width=False)

if __name__ == "__main__":
    main()

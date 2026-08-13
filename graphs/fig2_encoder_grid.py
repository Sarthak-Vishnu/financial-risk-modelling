# CAPTION: The encoder grid on the single forward split (train $<$ 2025, evaluate on
# 2025), with full-text windowed inputs. Points give the information coefficient and
# bars the bootstrap 95\% confidence interval. The dashed line marks the defended model,
# \texttt{struct+tfidf [sparse]}, and the dotted line the no-text structured baseline.
# A filled marker means the condition is significantly less accurate than the defended
# model under a Diebold--Mariano test. A hollow marker means the test cannot separate the
# two, which is a tie and not a win: no condition's interval lies clear of the dashed
# line, and every interval overlaps it.

"""Figure 2 -- the encoder grid as a point-and-interval plot (thesis Section 4.2).

Why a figure at all: the chapter's central negative claim is that nothing beats the
count model, and it currently arrives as a long table the reader has to scan and hold in
memory. A reference line turns that into one glance.

The intervals are the second reason. Section 4.1 states in prose that the interval spans
roughly +/- 0.06, which is why single-year differences of 0.01 to 0.02 can never be
conclusive. Drawing the intervals shows it instead of asserting it: every bar straddles
the reference line, so the plot makes the case for moving to multi-year paired tests
before the reader reaches Section 4.3.

Rows are sorted by IC, best at the top, so the reading is top-down.

NUMBERS ARE READ FROM THE COMMITTED GRID at phase5/out/stress_grid_val2025_fixed.json,
through graphs/_grid.py, and are not literals here. They were literals transcribed from
the write-up until the grid became a committed artefact; that is what let this figure
drift out of step with it. Which conditions belong in an encoder figure is still an
editorial choice and still lives in _grid.py, where an uncovered condition raises rather
than vanishing from the plot. R^2_log is deliberately not drawn: it is a third axis the
plot has no room for, and it survives in the appendix version of the table.
"""

import matplotlib.pyplot as plt

from _grid import REFERENCE, STRUCTURED, encoder_conditions, load_grid
from _style import C, TEXTWIDTH_IN, apply_style, save

ALPHA = 0.05

# The 'struct+' prefix is on most rows and carries no information.
# Set False to print the condition names exactly as the grid records them.
SHORT_LABELS = True

# Diagonal y-labels were tried at 25 degrees and made this figure worse, so the default
# is back to horizontal. The reason is geometry rather than taste: the longest label is
# about 3.2in of text, and rotating it by 25 degrees drops its far end 1.3in below its
# own tick, which is roughly four row heights. Every label then visually spans four rows
# and the reader cannot tell which row it belongs to. Tilting pays only when labels are
# short or rows are tall, and here neither holds. Set to 25 to see it.
LABEL_ROTATION = 0


def label_for(name):
    if SHORT_LABELS and name.startswith("struct+"):
        return name[len("struct+"):]
    return name


def main():
    apply_style()

    grid = load_grid()
    rows = sorted(encoder_conditions(grid), key=lambda r: r["ic"])  # best ends on top
    ref_ic = grid[REFERENCE]["ic"]
    base_ic = grid[STRUCTURED]["ic"]
    y = range(len(rows))

    # Height tracks the row count so the row pitch stays constant however many
    # conditions the grid carries. 0.245in per row plus fixed furniture reproduces the
    # pitch this figure was tuned at.
    height = 2.0 + 0.245 * len(rows)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, height))

    for i, r in zip(y, rows):
        # dm_p is None only for the reference row, which _grid.py excludes.
        worse = r["dm_p"] is not None and r["dm_p"] < ALPHA
        colour = C["vermillion"] if worse else C["blue"]
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=colour, linewidth=1.8,
                solid_capstyle="butt", zorder=2)
        ax.scatter([r["ic"]], [i], s=80, zorder=3, color=colour,
                   facecolors=colour if worse else "white",
                   edgecolors=colour, linewidths=1.8)

    ax.axvline(ref_ic, color=C["blue"], linestyle="--", linewidth=1.6, zorder=1)
    ax.axvline(base_ic, color=C["grey"], linestyle=":", linewidth=1.6, zorder=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels([label_for(r["name"]) for r in rows],
                       fontfamily="monospace", fontsize=10,
                       rotation=LABEL_ROTATION, ha="right", va="center",
                       rotation_mode="anchor")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Information coefficient (2025 validation year)")
    lo = min(min(r["ci_lo"] for r in rows), base_ic, ref_ic)
    hi = max(max(r["ci_hi"] for r in rows), base_ic, ref_ic)
    ax.set_xlim(lo - 0.02, hi + 0.02)
    ax.grid(axis="y", visible=False)

    # Reference lines named ABOVE the axes, at two heights so the two labels cannot
    # collide: the lines they mark are only about 0.012 apart.
    xt = ax.get_xaxis_transform()
    ax.text(base_ic, 1.11, STRUCTURED, transform=xt, color=C["grey"],
            fontsize=10.5, ha="center", va="bottom", fontfamily="monospace")
    ax.text(ref_ic, 1.02, REFERENCE, transform=xt, color=C["blue"],
            fontsize=10.5, ha="center", va="bottom", fontfamily="monospace")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=C["vermillion"],
                   label=rf"significantly worse, DM $p < {ALPHA:g}$"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor="white", markeredgecolor=C["blue"],
                   markeredgewidth=1.8, color=C["blue"],
                   label="tie: not separable from the reference"),
    ]
    # bbox_to_anchor is in axes fractions, so a fixed -0.32 would move the legend further
    # from the axes in inches as the figure grows. Scale it to hold the gap constant at
    # what -0.32 gave on the 5.4in canvas this was tuned on.
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.32 * 5.4 / height), ncol=1)

    save(fig, "fig2_encoder_grid", exact_width=False)


if __name__ == "__main__":
    main()

# CAPTION: The change in IC from adding earnings-call tone to the structured model, at each
# of the eight measurement windows and at both anchors. Hollow markers rest on 139 filings
# across six quarters rather than the 152 across seven behind the filled markers.
#
# Captions stay descriptive: what the figure shows, what the notation means, and anything
# needed to read it. The finding it supports belongs in the prose beside it, not here.
"""Figure 5 -- the call-tone window sweep, as a panel beside its anchor table.

Why a figure rather than the eight-column table it replaces: the point of these two
series is their shape against each other, one flat and one humped, and a row of eight
numbers makes the reader reconstruct that. It also has to sit in half a text width beside
the anchor table, where an eight-column table does not fit at a legible size.

Same reduced-sample caveat as fig3_three_peaks: the sixty- and ninety-day call-anchor
points rest on a smaller sample, and they are also the two that turn negative, so the
sample change and the sign change coincide.

NUMBERS ARE EMBEDDED LITERALS, from the earnings-call table of the write-up.
"""

import matplotlib.pyplot as plt

from _style import C, TEXTWIDTH_IN, apply_style, save, window_axis, window_positions

WIDTH_IN = TEXTWIDTH_IN
NAME = "fig5_call_windows"
FONT_SCALE = 1.0
LEGEND_SCALE = 1.0

WINDOWS = [3, 5, 7, 10, 20, 30, 60, 90]
FILING = [0.011, 0.006, -0.003, -0.001, -0.006, -0.005, 0.002, -0.003]
CALL = [-0.034, 0.079, 0.084, 0.142, 0.134, 0.039, -0.028, -0.025]
CALL_REDUCED = {60, 90}          # n=139 over 6 quarters, not 152 over 7


def main():
    apply_style()
    if FONT_SCALE != 1.0:
        import matplotlib as mpl
        for k in ("font.size", "axes.labelsize", "xtick.labelsize",
                  "ytick.labelsize", "legend.fontsize"):
            mpl.rcParams[k] = mpl.rcParams[k] * FONT_SCALE
        mpl.rcParams["legend.fontsize"] *= LEGEND_SCALE
        mpl.rcParams["lines.markersize"] *= FONT_SCALE

    X = window_positions(WINDOWS)
    fig, ax = plt.subplots(
        figsize=(WIDTH_IN, 3.6 * min(1.0, WIDTH_IN / TEXTWIDTH_IN) + 0.9),
        layout="constrained")

    ax.axhline(0, color=C["grey"], linewidth=1.0, zorder=1)
    ax.plot(X, CALL, color=C["vermillion"], marker="o", zorder=3, label="call anchor")
    ax.plot(X, FILING, color=C["blue"], marker="s", zorder=3, label="filing anchor")

    # redraw the reduced-sample call points hollow, over the line
    part = [(x, v) for x, w, v in zip(X, WINDOWS, CALL) if w in CALL_REDUCED]
    ax.scatter([x for x, _ in part], [v for _, v in part], s=55, facecolors="white",
               edgecolors=C["vermillion"], linewidths=1.8, zorder=4)

    ax.set_ylabel(r"$\Delta$IC over structured")
    ax.set_xlabel("Measurement window (days)")
    ax.legend(loc="upper right", handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
    window_axis(ax, WINDOWS)
    save(fig, NAME)


if __name__ == "__main__":
    main()

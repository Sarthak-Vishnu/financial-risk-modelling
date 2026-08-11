# CAPTION: The text increment across measurement windows, on the 2018--2024
# expanding-window backtest. The upper panel gives the information coefficient of the
# three models at each of the eight trading windows. The lower panel gives the increment
# that text adds, measured as \texttt{struct+tfidf [sparse]} minus
# \texttt{structured [ridge]} on the same firms. Filled markers are windows where that
# increment reaches conventional significance, hollow markers are windows where it does
# not. The three models keep the same order at every window. The increment itself is
# largest at seven days and falls to zero by sixty, even though all three models predict
# better as the window gets longer.

"""Figure 1 -- the term structure of the text increment (thesis Section 4.3.4).

Why a figure at all: Section 4.3.4 states three findings and every one of them is a
statement about shape. The ranking holds at every window (upper panel, no crossings),
the increment is single-peaked rather than monotonic (lower panel), and significance
appears at the short end (filled markers). A reader currently reconstructs all three
from a six-column table.

The upper panel is doing more than setting context. It carries the explanation the
prose has to spell out: predictability climbs as the window lengthens, because longer
realised-volatility windows are smoother and increasingly dominated by the persistent
component the structured block already measures. That is why the increment falls even
as every curve rises.

NUMBERS ARE EMBEDDED LITERALS, taken from Table 4.5 (tab:horizon) of thesis.tex. They
are not recomputed here. If the table changes, change them here too: a figure that
silently disagrees with its own table is worse than no figure.
"""

import matplotlib.pyplot as plt

from _style import (C, TEXTWIDTH_IN, apply_style, save, window_axis,
                    window_positions)

# ---- Table 4.5 (tab:horizon) ------------------------------------------------
WINDOWS = [3, 5, 7, 10, 20, 30, 60, 90]
LAGGED = [0.193, 0.289, 0.299, 0.314, 0.401, 0.461, 0.528, 0.534]
STRUCTURED = [0.330, 0.417, 0.444, 0.462, 0.513, 0.546, 0.592, 0.607]
STRUCT_TFIDF = [0.351, 0.441, 0.470, 0.482, 0.532, 0.561, 0.591, 0.602]
DELTA = [0.021, 0.023, 0.027, 0.020, 0.020, 0.016, -0.001, -0.005]
PVALUE = [0.020, 0.017, 0.014, 0.076, 0.119, 0.098, 0.851, 0.477]

ALPHA = 0.05


def main():
    apply_style()
    X = window_positions(WINDOWS)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(TEXTWIDTH_IN, 5.8), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.get_layout_engine().set(hspace=0.02, h_pad=0.02, w_pad=0.02)

    # ---- upper: the three lanes ----
    ax1.plot(X, STRUCT_TFIDF, marker="o", color=C["blue"],
             label=r"struct+tfidf [sparse]")
    ax1.plot(X, STRUCTURED, marker="s", color=C["vermillion"],
             label=r"structured [ridge]")
    ax1.plot(X, LAGGED, marker="^", color=C["grey"],
             label=r"lagged [hgb]")
    ax1.set_ylabel("Information coefficient")
    ax1.set_ylim(0.15, 0.66)
    ax1.legend(loc="lower right", ncol=1, handlelength=2.4)

    # ---- lower: the increment ----
    ax2.axhline(0, color=C["grey"], linewidth=1.0, zorder=1)
    ax2.plot(X, DELTA, color=C["blue"], zorder=2)

    sig = [i for i, p in enumerate(PVALUE) if p < ALPHA]
    non = [i for i, p in enumerate(PVALUE) if p >= ALPHA]
    ax2.scatter([X[i] for i in sig], [DELTA[i] for i in sig],
                s=70, facecolors=C["blue"], edgecolors=C["blue"], zorder=3,
                label=rf"$p < {ALPHA:g}$")
    ax2.scatter([X[i] for i in non], [DELTA[i] for i in non],
                s=70, facecolors="white", edgecolors=C["blue"], linewidths=1.8,
                zorder=3, label=rf"$p \geq {ALPHA:g}$")

    # mark the peak, which is the point Section 4.3.4 turns on
    peak = max(range(len(DELTA)), key=lambda i: DELTA[i])
    ax2.annotate(f"peak {DELTA[peak]:+.3f}\nat {WINDOWS[peak]} days",
                 xy=(X[peak], DELTA[peak]),
                 xytext=(X[peak] + 0.45, DELTA[peak] + 0.004),
                 fontsize=11, color=C["blue"],
                 arrowprops=dict(arrowstyle="-", color=C["blue"], linewidth=1.0))

    # Naming the two models on the axis, not only in the caption: the lower panel is a
    # difference, and a difference is meaningless until the reader knows of what.
    # Three short lines, not one long one. A rotated label's text length becomes its
    # VERTICAL extent, so "struct+tfidf vs structured" on one line is taller than the
    # panel it labels and spills past the figure edge.
    ax2.set_ylabel("Text $\\Delta$IC\nstruct+tfidf\n$-$ structured", fontsize=10)
    ax2.set_xlabel("Measurement window (trading days)")
    ax2.set_ylim(-0.011, 0.037)
    ax2.legend(loc="lower left", ncol=2, handletextpad=0.3, columnspacing=1.2)

    window_axis(ax2, WINDOWS)
    save(fig, "fig1_window_spectrum")


if __name__ == "__main__":
    main()

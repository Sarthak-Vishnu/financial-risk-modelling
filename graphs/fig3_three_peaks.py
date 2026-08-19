# CAPTION: Where each effect peaks across the measurement-window spectrum. Each panel
# carries its own vertical scale, because the three effects differ in magnitude by roughly
# an order of magnitude and the comparison of interest is where each peaks rather than how
# high. The filing's own text peaks earliest, at seven days. Both event families peak
# later, call tone at ten days and insider disagreement at thirty, and both are negative at
# the shortest window of three days. Were the shared mechanism simply that more text
# signal is better, the three would peak together. In the middle panel, the hollow markers
# at sixty and ninety days represents they were measured differently. They rest on 139 
# filings over six quarters rather than the 152 over seven quarters behind the filled 
# markers, because a forward window that long runs past the end of the price data for the
# final quarter's calls.

import matplotlib.pyplot as plt

from _style import (C, TEXTWIDTH_IN, apply_style, save, window_axis,
                    window_positions)

WINDOWS = [3, 5, 7, 10, 20, 30, 60, 90]

# ---- Series 1: 10-K text increment. Table 4.5 (tab:horizon). ---------------
TENK = [0.021, 0.023, 0.027, 0.020, 0.020, 0.016, -0.001, -0.005]

# ---- Series 2: call tone at the call anchor, HGB lane, 2025. ---------------
CALL_ANCHOR = [-0.034, 0.079, 0.084, 0.142, 0.134, 0.039, -0.028, -0.025]
CALL_REDUCED_SAMPLE = {60, 90}          # n=139 over 6 quarters, not 152 over 7

# ---- Series 3: insider-disagreement conditioning, 2018-2024 backtest. ------
DISAGREEMENT = [-0.038, -0.011, -0.002, 0.002, 0.029, 0.030, 0.001, 0.002]

PANELS = [
    ("10-K text increment",     TENK,         C["blue"],       set()),
    ("Call tone, call anchor",  CALL_ANCHOR,  C["vermillion"], CALL_REDUCED_SAMPLE),
    ("Insider disagreement",    DISAGREEMENT, C["green"],      set()),
]

def main():
    apply_style()
    X = window_positions(WINDOWS)
    fig, axes = plt.subplots(
        3, 1, figsize=(TEXTWIDTH_IN, 6.4), sharex=True, layout="constrained")
    fig.get_layout_engine().set(hspace=0.04, h_pad=0.02, w_pad=0.02)

    for ax, (name, values, colour, reduced) in zip(axes, PANELS):
        ax.axhline(0, color=C["grey"], linewidth=1.0, zorder=1)
        ax.plot(X, values, color=colour, linewidth=2.2, zorder=2)

        full = [(x, v) for x, w, v in zip(X, WINDOWS, values) if w not in reduced]
        ax.scatter([w for w, _ in full], [v for _, v in full],
                   s=75, color=colour, zorder=4)
        if reduced:
            part = [(x, v) for x, w, v in zip(X, WINDOWS, values) if w in reduced]
            ax.scatter([w for w, _ in part], [v for _, v in part],
                       s=75, facecolors="white", edgecolors=colour, linewidths=2.0,
                       zorder=4)

        # The dotted line, not a leader arrow, is what marks the peak. Anchoring the
        # text to the point itself collides with the panel label whenever the peak sits
        # at the short end, which it does in two panels out of three.
        px, pw, pv = max(zip(X, WINDOWS, values), key=lambda t: t[2])
        ax.axvline(px, color=colour, linestyle=":", linewidth=1.6, alpha=0.85, zorder=1)
        ax.text(0.988, 0.92, f"peak {pv:+.3f} at {pw} days", transform=ax.transAxes,
                fontsize=11.5, color=colour, ha="right", va="top")

        ax.set_ylabel(r"$\Delta$IC", labelpad=2)
        ax.text(0.012, 0.92, name, transform=ax.transAxes, fontsize=12,
                color=colour, ha="left", va="top")
        span = list(values) + [0.0]
        pad = max(0.006, 0.22 * (max(span) - min(span)))
        ax.set_ylim(min(span) - pad, max(span) + pad * 2.6)

    # No legend for the hollow markers: the only free space in the middle panel is
    # exactly where those two points sit, so a legend collides with what it explains.
    # The caption carries it instead.
    axes[-1].set_xlabel("Measurement window (trading days)")
    window_axis(axes[-1], WINDOWS)
    save(fig, "fig3_three_peaks")

if __name__ == "__main__":
    main()

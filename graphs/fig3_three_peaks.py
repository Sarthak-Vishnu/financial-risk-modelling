# CAPTION: Where each effect peaks across the measurement-window spectrum. Each panel 
# carries its own vertical scale, because the three effects differ in magnitude by 
# roughly an order of magnitude and the comparison of interest is where each peaks 
# rather than how high. The filing's own text peaks at seven days. Both event families 
# peak over a window of weeks and are absent or inverted at three days. Were the shared
# mechanism simply that more text signal is better, the three would peak together.

"""Figure 3 -- the three effects do not peak over the same window (thesis Section 4.6.1).

Why a figure at all: this is the most interesting result in the chapter and the only
one with no exhibit. The argument in Section 4.6.1 is explicitly about shape, and it
ends "Were the shared mechanism simply that more text signal is better, all three
curves would peak together. They do not." Right now the reader has to assemble three
series from prose scattered across three sections and compare them mentally.

WHY THREE PANELS AND NOT ONE. The call-anchored effect reaches +0.142 while the 10-K
increment tops out at +0.027. On a shared axis the 10-K curve is a flat line and the
figure argues the opposite of what the text says. Separate panels with separate scales
are the honest presentation, because the claim is about the location of each peak, not
its height. Normalising each series to its own maximum would also work, but several
values are negative, which makes that scaling hard to read.

DATA STATUS: incomplete. The 10-K series is complete. The call-anchor series has five
of eight values recorded in Section 4.5; the other three are described only as
"negative". The insider-disagreement series has three of eight recorded. Run the script
and it prints exactly which cells are missing. Nothing is interpolated or invented: a
missing value is drawn as a gap, and the connecting segment across a gap is dashed and
faint so it can never be mistaken for a measurement.
"""

import matplotlib.pyplot as plt

from _style import C, TEXTWIDTH_IN, apply_style, save, window_axis

WINDOWS = [3, 5, 7, 10, 20, 30, 60, 90]

# ---- Series 1: 10-K text increment. Table 4.5 (tab:horizon). Complete. ------
TENK = [0.021, 0.023, 0.027, 0.020, 0.020, 0.016, -0.001, -0.005]

# ---- Series 2: call tone at the call anchor. Section 4.5 prose, n = 152. ----
# "negative at three days, then +0.079, +0.084, +0.142 and +0.134 at five, seven,
#  ten and twenty days, +0.039 at thirty, and negative again at sixty and ninety."
# TODO: the three None cells have a recorded sign but no recorded value.
CALL_ANCHOR = [None, 0.079, 0.084, 0.142, 0.134, 0.039, None, None]

# ---- Series 3: insider-disagreement conditioning. 2018-2024 backtest. -------
# Section 4.6.1 gives -0.038 at three days; Section 4.6 gives +0.029 at twenty and
# +0.030 at thirty.
# TODO: five cells are not recorded anywhere in the thesis or study_extended.md.
DISAGREEMENT = [-0.038, None, None, None, 0.029, 0.030, None, None]

PANELS = [
    ("10-K text increment",     TENK,         C["blue"]),
    ("Call tone, call anchor",  CALL_ANCHOR,  C["vermillion"]),
    ("Insider disagreement",    DISAGREEMENT, C["green"]),
]


def plot_with_gaps(ax, values, colour):
    """Solid between adjacent measured windows, faint dashed across a gap.

    The distinction matters. A solid line through a gap asserts a measurement that was
    never taken."""
    known = [(w, v) for w, v in zip(WINDOWS, values) if v is not None]
    if len(known) > 1:
        ax.plot([w for w, _ in known], [v for _, v in known],
                color=colour, linestyle="--", linewidth=1.1, alpha=0.45, zorder=2)
    for i in range(len(WINDOWS) - 1):
        if values[i] is not None and values[i + 1] is not None:
            ax.plot(WINDOWS[i:i + 2], values[i:i + 2],
                    color=colour, linewidth=2.2, zorder=3)
    ax.scatter([w for w, _ in known], [v for _, v in known],
               s=75, color=colour, zorder=4)
    return known


def main():
    apply_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(TEXTWIDTH_IN, 6.4), sharex=True,
        gridspec_kw={"hspace": 0.18})

    missing = []
    for ax, (name, values, colour) in zip(axes, PANELS):
        ax.axhline(0, color=C["grey"], linewidth=1.0, zorder=1)
        known = plot_with_gaps(ax, values, colour)

        if known:
            # The dotted line, not a leader arrow, is what marks the peak. Anchoring
            # the text to the point itself collides with the panel label whenever the
            # peak sits at the short end, which it does in two panels out of three.
            pw, pv = max(known, key=lambda t: t[1])
            ax.axvline(pw, color=colour, linestyle=":", linewidth=1.6, alpha=0.85,
                       zorder=1)
            ax.text(0.988, 0.92, f"peak {pv:+.3f} at {pw} days",
                    transform=ax.transAxes, fontsize=11.5, color=colour,
                    ha="right", va="top")

        gaps = [w for w, v in zip(WINDOWS, values) if v is None]
        missing.extend((name, w) for w in gaps)

        ax.set_ylabel(r"$\Delta$IC", labelpad=2)
        ax.text(0.012, 0.92, name, transform=ax.transAxes, fontsize=12,
                color=colour, ha="left", va="top")
        span = [v for _, v in known] + [0.0]
        pad = max(0.006, 0.22 * (max(span) - min(span)))
        ax.set_ylim(min(span) - pad, max(span) + pad * 2.6)

    axes[-1].set_xlabel("Measurement window (trading days, log scale)")
    window_axis(axes[-1], WINDOWS)
    save(fig, "fig3_three_peaks")

    if missing:
        print("\nMISSING CELLS -- the figure has gaps until these are supplied:")
        for name, w in missing:
            print(f"  {name:<36s} window = {w:>2d} days")
        print("Fill them in the series constants at the top of this file.")


if __name__ == "__main__":
    main()

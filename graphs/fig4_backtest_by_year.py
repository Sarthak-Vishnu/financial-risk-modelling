# CAPTION: Year-by-year information coefficient across the expanding-window backtest. Each
# point is a model trained on all filings strictly dated before that year and scored on it,
# so the series is what a deployed model would have produced. The shaded band marks 2020, 
# whose regime break dominates the year-to-year variance of the thirty-day panel and is the
# reason the increment's significance resolves by shortening the measurement window rather
# than by accumulating further evaluation years.

"""Figure 4 -- the backtest year by year (thesis Section 4.3.1).

Why a figure at all: Table 4.3 reports a mean IC and a t-statistic, and the prose says
the t-statistic "measures the consistency of the ranking skill". Consistency is a
property of a series, and the series is never shown. This figure shows it.

It also supports a claim the chapter leans on twice with no visual backing: that 2020 is
a regime break large enough to dominate the year-to-year variance of the thirty-day
panel. Section 4.3.4 uses exactly that to explain why the text increment reaches
significance by shortening the window rather than by adding evaluation years.

DATA STATUS: not available. The per-year series is in the cluster logs and is not reproduced here. Paste the
yearly ICs into SERIES below and the script will run.

The check_against_table_43 guard is the reason this is worth doing carefully rather
than quickly. It recomputes the mean and the t-statistic from whatever series you paste
and compares them against the published Table 4.3 values. A transcription slip in
twelve hand-copied numbers is close to undetectable by eye, and would put a figure into
the thesis that contradicts a table three pages earlier.
"""

import math

import matplotlib.pyplot as plt

from _style import C, TEXTWIDTH_IN, apply_style, save

# Set to a fraction of the text width to render for a side-by-side layout.
# Regenerating at the target width is not the same as scaling the full-width
# file down in LaTeX: scaling shrinks the fonts with everything else, whereas
# this lays the axis out for the box it will sit in.
WIDTH_IN = TEXTWIDTH_IN
NAME = "fig4_backtest_by_year"
# Fonts are set for a full-width figure. In a half-width panel the same point sizes
# are oversized relative to the axes, so scale them down and the legend further still,
# which is what lets it sit inside the plot under the curve rather than below it.
FONT_SCALE = 1.0
LEGEND_SCALE = 1.0

YEARS = list(range(2013, 2025))          # 2013 to 2024 inclusive

# ---- Per-year IC, from the backtest logs. One value per year in YEARS. ------
# struct+tfidf 2020 is a real +0.00011923, not a missing value. 2020 is the year all
# three models collapse, and persistence alone goes outright negative.
SERIES = {
    "struct+tfidf [sparse]": [0.714, 0.668, 0.561, 0.773, 0.719, 0.582,
                              0.665, 0.000, 0.604, 0.750, 0.655, 0.673],
    "structured [ridge]":    [0.714, 0.670, 0.549, 0.771, 0.715, 0.574,
                              0.657, -0.025, 0.584, 0.749, 0.598, 0.682],
    "lagged [hgb]":          [0.687, 0.649, 0.461, 0.679, 0.635, 0.484,
                              0.543, -0.121, 0.650, 0.582, 0.485, 0.571],
}

STYLE = {
    "struct+tfidf [sparse]": (C["blue"], "o"),
    "structured [ridge]":    (C["vermillion"], "s"),
    "lagged [hgb]":          (C["green"], "^"),
}

# ---- Table 4.3 (tab:backtest): (2018-2024 IC, t), (2013-2024 IC, t) ----------
TABLE_43 = {
    "lagged [hgb]":          ((0.456, 4.6), (0.526, 8.3)),
    "structured [ridge]":    ((0.546, 5.6), (0.603, 9.9)),
    "struct+tfidf [sparse]": ((0.561, 5.9), (0.614, 10.4)),
}

REGIME_YEAR = 2020


def mean_and_t(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, mean / (math.sqrt(var / n))


def check_against_table_43(lane, values):
    """Recompute Table 4.3 from the pasted series. Warn loudly on disagreement."""
    if lane not in TABLE_43:
        return
    windows = {"2018-2024": [v for y, v in zip(YEARS, values) if y >= 2018],
               "2013-2024": list(values)}
    for (label, subset), (ic_ref, t_ref) in zip(windows.items(), TABLE_43[lane]):
        ic, t = mean_and_t(subset)
        ok = abs(ic - ic_ref) <= 0.0015 and abs(t - t_ref) <= 0.15
        flag = "ok " if ok else "MISMATCH"
        print(f"  [{flag}] {lane:<24s} {label}  "
              f"IC {ic:.3f} (table {ic_ref:.3f})   t {t:.1f} (table {t_ref:.1f})")


def main():
    populated = {k: v for k, v in SERIES.items() if v}
    if not populated:
        raise SystemExit(
            "No per-year data. Paste the yearly ICs into SERIES at the top of this "
            "file, one value for each year in YEARS (2013 to 2024), then rerun.")

    print("Checking the pasted series against Table 4.3:")
    for lane, values in populated.items():
        if len(values) != len(YEARS):
            raise SystemExit(f"{lane}: got {len(values)} values, expected {len(YEARS)}")
        check_against_table_43(lane, values)

    apply_style()
    if FONT_SCALE != 1.0:
        import matplotlib as mpl
        for k in ("font.size", "axes.labelsize", "xtick.labelsize",
                  "ytick.labelsize", "legend.fontsize"):
            mpl.rcParams[k] = mpl.rcParams[k] * FONT_SCALE
        mpl.rcParams["legend.fontsize"] *= LEGEND_SCALE
        mpl.rcParams["lines.markersize"] *= FONT_SCALE
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.9 * min(1.0, WIDTH_IN / TEXTWIDTH_IN) + 0.9),
                           layout="constrained")

    ax.axvspan(REGIME_YEAR - 0.5, REGIME_YEAR + 0.5, color=C["faint"], alpha=0.35,
               zorder=0, linewidth=0)
    # The defended model is drawn last so it sits above the other two wherever the
    # three cross, which is most of the panel.
    order = ["lagged [hgb]", "structured [ridge]", "struct+tfidf [sparse]"]
    for z, lane in enumerate(k for k in order if k in populated):
        colour, marker = STYLE.get(lane, (C["green"], "D"))
        ax.plot(YEARS, populated[lane], marker=marker, color=colour, label=lane,
                zorder=2 + z)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", handletextpad=0.4,
              borderpad=0.3, labelspacing=0.3, borderaxespad=0.4)

    ax.set_xlabel("Test year")
    ax.set_ylabel("Information coefficient")
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha="right")
    # A narrow panel has no interior space that the series do not cross, so the
    # legend goes under the axes there rather than on top of the data.
    save(fig, NAME)


if __name__ == "__main__":
    main()

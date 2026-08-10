# CAPTION: Year-by-year information coefficient across the expanding-window backtest. Each point is a model trained on all filings strictly dated before that year and scored on it, so the series is what a deployed model would have produced. The shaded band marks 2020, whose regime break dominates the year-to-year variance of the thirty-day panel and is the reason the increment's significance resolves by shortening the measurement window rather than by accumulating further evaluation years.
"""Figure 4 -- the backtest year by year (thesis Section 4.3.1).

Why a figure at all: Table 4.3 reports a mean IC and a t-statistic, and the prose says
the t-statistic "measures the consistency of the ranking skill". Consistency is a
property of a series, and the series is never shown. This figure shows it.

It also supports a claim the chapter leans on twice with no visual backing: that 2020 is
a regime break large enough to dominate the year-to-year variance of the thirty-day
panel. Section 4.3.4 uses exactly that to explain why the text increment reaches
significance by shortening the window rather than by adding evaluation years.

DATA STATUS: not available. The per-year series is in the cluster logs and appears
neither in thesis.tex nor in study_extended.md, so nothing is filled in here. Paste the
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

YEARS = list(range(2013, 2025))          # 2013 to 2024 inclusive

# ---- TODO: paste the per-year IC for each lane, one value per year in YEARS ----
# Leave a lane as an empty list to omit it from the figure.
SERIES = {
    "struct+tfidf [sparse]": [],
    "structured [ridge]":    [],
    "lagged [hgb]":          [],
}

STYLE = {
    "struct+tfidf [sparse]": (C["blue"], "o"),
    "structured [ridge]":    (C["vermillion"], "s"),
    "lagged [hgb]":          (C["grey"], "^"),
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
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.9))

    ax.axvspan(REGIME_YEAR - 0.5, REGIME_YEAR + 0.5, color=C["faint"], alpha=0.35,
               zorder=0, linewidth=0)
    for lane, values in populated.items():
        colour, marker = STYLE.get(lane, (C["green"], "D"))
        ax.plot(YEARS, values, marker=marker, color=colour, label=lane, zorder=2)

    ax.set_xlabel("Test year")
    ax.set_ylabel("Information coefficient")
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha="right")
    ax.legend(loc="lower left", ncol=1)
    save(fig, "fig4_backtest_by_year")


if __name__ == "__main__":
    main()

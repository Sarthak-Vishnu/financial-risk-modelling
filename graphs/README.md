# Chapter 4 result figures

Figures for the Results chapter of the thesis. Each script writes a vector PDF for
inclusion and a PNG for checking the result quickly.

```bash
cd graphs
python fig1_window_spectrum.py
python fig3_three_peaks.py
python fig4_backtest_by_year.py
```

Output lands in `graphs/out/`. Copy the PDFs into `S2880814_MScDiss/images/`.

## What is in the thesis, and what is not

| Script | In the thesis | Source of the numbers |
|---|---|---|
| `fig1_window_spectrum.py` | **Figure 4.2**, §4.3.4 | Table A.1 (`tab:horizon`) |
| `fig3_three_peaks.py` | **Figure 4.3**, §4.6.1 | Table A.1, plus the call-anchor and insider-conditioning window sweeps from the result logs |
| `fig4_backtest_by_year.py` | **Figure 4.1**, §4.3.1 | Per-year backtest logs, checked against Table 4.3 |
| `fig2_encoder_grid.py` | **not used** | `phase5/out/stress_grid_val2025_fixed.json`, read at run time |
| `fig2_extra_delta_by_family.py` | **not used** | as above |

Both fig2 variants were built and rejected: Table 4.2 was kept in the main text instead,
because the table carries `R^2_log` and the exact confidence intervals that a
point-and-interval plot has no room for, and the chapter did not need a second exhibit
making the same negative point. They are kept in the repository rather than deleted,
since the argument for a figure there may come back if that table is ever demoted to the
appendix.

**Both now read the committed grid directly**, through `_grid.py`, rather than carrying
transcribed literals. They used to carry literals from Table 4.2, and that is exactly how
they went stale: the grid was regenerated on corrected data, gained four rows for the
project's own three-view encoders, and two conditions changed name because the encoder
they select is chosen within-sample. The figures kept plotting the superseded numbers and
nothing flagged it. Reading the JSON means they cannot disagree with it, and it removes
the dependence on a table this repository cannot see. The two references they draw — the
defended model's IC and the single-year noise band — come from the same file, because
`struct+tfidf [sparse]` is itself a row of the grid: the level is that row's IC and the
band is that row's own bootstrap interval, so the band is the measured 0.068 rather than
the rounded 0.06 the text quotes.

What stays editorial in `_grid.py` is which conditions belong in a figure about encoders,
and which family each encoder falls into. Neither is recorded in the run. A condition in
the JSON that no rule covers raises rather than being skipped, so a new encoder entering
`stress_grid.py` surfaces as a failure here instead of as a row quietly missing from a
plot.

`fig1_window_spectrum.py` replaced Table A.1 in the main text. That table moved to
Appendix A, where it still carries the exact *p*-values and the validation-year column
the figure does not show.

## Where the numbers come from

**The three thesis figures carry embedded literals.** Nothing is recomputed from the
datasets, for one reason: a figure that recomputes can silently disagree with the table
printed three pages earlier. Each script names its source in the docstring. If a table
changes, change the constants too.

That reasoning does not extend to a result that is itself committed. The two fig2 variants
read `phase5/out/stress_grid_val2025_fixed.json` at run time, because the grid is a
tracked file rather than a number living only in the write-up, so there is a single source
to agree with. The other three have no committed equivalent to read.

`fig4_backtest_by_year.py` checks itself, because its twelve hand-copied values per lane
are exactly the situation where a transcription slip goes unnoticed. It recomputes the
mean IC and the *t*-statistic over both the 2013–2024 and 2018–2024 sub-windows and
compares them against the published Table 4.3 values, printing `MISMATCH` on any
disagreement. All six currently reproduce.

Two cells in `fig3_three_peaks.py` are drawn hollow rather than filled. The sixty- and
ninety-day call-anchor points rest on 139 filings across six quarters rather than the 152
across seven behind the other six, because a forward window that long runs past the end
of the price data for the final quarter's calls. Those two are also the two that turn
negative, so the sample change and the sign change coincide, and drawing all eight
identically would invite a reader to attribute to the window something that may belong to
the sample.

## Conventions

Set once in `_style.py`, so changing them changes every figure.

**Width is fixed, not cropped.** `infthesis` loads geometry with `a4paper`, `left=4cm`,
`right=2.5cm`, so `\textwidth` is 14.5cm, or 411.0pt. Figures are written at exactly that,
so `\includegraphics[width=\textwidth]` applies no scaling and the point sizes in
`_style.py` are the sizes that land on the page.

This is worth stating because the first version got it wrong in a way that was invisible
until the figures sat in the same chapter. `bbox_inches="tight"` crops to content, so
three figures with different label widths came out at 390.6, 378.2 and 363.3pt, and
`width=\textwidth` then stretched each by 5%, 9% and 13%. The same 12pt label rendered at
three different sizes. Figures for the thesis now use constrained layout and save at the
nominal canvas size; `save(..., exact_width=False)` falls back to the cropped save, which
is what the two unused fig2 variants need, since both place a legend outside the axes.

**Fixing the canvas means content can overflow it**, which a cropped save silently
absorbed. `save()` measures the rendered content against the canvas afterwards and warns.
The usual culprit is a rotated multi-line y-label, whose text length becomes its vertical
extent and which can end up taller than the panel it labels.

**Measurement windows are spaced evenly, not by length.** Linear is unusable, since 3, 5,
7 and 10 days would take the first eighth of the axis and that is where the structure is.
Log was tried and is worse than it looks, because it encodes ratio rather than elapsed
time: under log the step from 10 to 20 days is drawn wider than the step from 60 to 90,
although the first spans ten days and the second thirty. Even spacing claims nothing about
the gaps, and the tick labels carry the real values.

**No figure carries a title.** The LaTeX caption does that job. A suggested caption is the
first line of each script, as a comment, ready to paste. Keep captions descriptive:
findings belong in the prose beside the figure, and the caption should carry only what a
reader needs in order to read the exhibit.

**Fonts are large on purpose.** These are read on screen at `\textwidth` in a 12pt
document, where anything under 11pt disappears.

**Colour carries meaning on its own**, since submission is digital. The palette is
Okabe-Ito, which costs nothing and stays legible under colour-vision deficiency.

## Using one in the thesis

```latex
\begin{figure}[t!]
    \centering
    \includegraphics[width=\textwidth]{images/fig1_window_spectrum.pdf}
    \caption{<paste the caption from line 1 of the script>}
    \label{fig:horizon}
\end{figure}
```

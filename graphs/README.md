# Chapter 4 result figures

Figures for the Results chapter of the thesis. Each script writes a vector PDF for
inclusion and a PNG for checking the result quickly.

```bash
cd graphs
python fig1_window_spectrum.py
python fig2_encoder_grid.py
python fig3_three_peaks.py
python fig4_backtest_by_year.py     # needs data pasted in first, see below
```

Output lands in `graphs/out/`. Copy the PDFs into `S2880814_MScDiss/images/`.

## Where the numbers come from

**Every number is an embedded literal taken from a table in `thesis.tex`.** Nothing is
recomputed from the datasets, for one reason: a figure that recomputes can silently
disagree with the table printed three pages earlier. Each script names its source table
in the docstring. If a table changes, change the constants too.

| Script | Source | Status |
|---|---|---|
| `fig1_window_spectrum.py` | Table 4.5 (`tab:horizon`) | complete |
| `fig2_encoder_grid.py` | Table 4.2 (`tab:encgrid`), reference lines from Table 4.1 | complete |
| `fig3_three_peaks.py` | Table 4.5, plus §4.5 and §4.6.1 prose | **incomplete, see below** |
| `fig4_backtest_by_year.py` | not recorded anywhere | **needs data, see below** |

## What is missing

**`fig3_three_peaks.py`** draws three series and two of them have gaps. Run it and it
prints exactly which cells are missing. Currently:

- Call tone at the call anchor: 3, 60 and 90 days. §4.5 records only that these are
  negative, not their values.
- Insider disagreement: 5, 7, 10, 60 and 90 days. Only 3, 20 and 30 days are recorded
  anywhere.

Missing cells are drawn as gaps. The faint dashed segment that bridges a gap is
deliberately distinguishable from a measured segment, since a solid line across a gap
asserts a measurement that was never taken.

**`fig4_backtest_by_year.py`** has no data at all. The per-year IC series lives in the
cluster logs and appears neither in `thesis.tex` nor in `study_extended.md`. Paste the
twelve yearly values per lane into `SERIES` and rerun.

That script then checks itself: it recomputes the mean IC and the *t*-statistic over
both the 2013–2024 and 2018–2024 sub-windows and compares them against the published
Table 4.3 values, printing `MISMATCH` on any disagreement. Twelve hand-copied numbers
per lane is exactly the situation where a transcription slip goes unnoticed and puts a
figure into the thesis that contradicts a table.

## Conventions

Set once in `_style.py`, so changing them changes every figure.

**Width is measured, not guessed.** `infthesis` loads geometry with `a4paper`,
`left=4cm`, `right=2.5cm`, so `\textwidth` is 14.5cm, or 5.709in. Figures are drawn at
exactly that, which means `\includegraphics[width=\textwidth]` applies no scaling and
the point sizes in `_style.py` are the point sizes that land on the page. Draw at any
other width and scaling silently changes every font size in the figure.

**No figure carries a title.** The LaTeX caption does that job. A suggested caption is
the first line of each script, as a comment, ready to paste.

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

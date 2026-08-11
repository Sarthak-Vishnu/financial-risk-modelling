# CAPTION: Every encoder configuration measured as a difference from the defended model,
# \texttt{struct+tfidf [sparse]}, on the 2025 validation year, grouped by what the encoder
# was trained to do. The shaded band is the width of a single-year bootstrap confidence
# interval, roughly $\pm 0.06$ at $n = 393$. A filled marker is significantly less
# accurate than the defended model under a Diebold--Mariano test; a hollow marker cannot
# be separated from it. Every configuration falls inside the band, so no difference in the
# grid is resolvable on one year of data. The three configurations that exceed zero do so
# by at most $0.005$, and none of them significantly.

"""Figure 2, alternative -- the encoder grid as differences from the defended model.

This is a second take on the same numbers as fig2_encoder_grid.py, kept alongside it
rather than replacing it. Two things it does that the sorted forest plot does not.

FIRST, IT GROUPS BY WHAT THE ENCODER WAS TRAINED TO DO. Section 4.2 concludes "task
alignment, not model modernity, closes the gap", and a plot sorted purely by IC hides
exactly that: the reader cannot see that sbert, bge and dual are one kind of thing and
ftvol and volaware another. Grouping puts the chapter's actual argument on the y-axis.

SECOND, IT PUTS THE DIFFERENCE ON THE X-AXIS RATHER THAN THE LEVEL. The question is not
what each condition scores, it is how each compares to the count model, and a reader
should not have to do fourteen subtractions by eye. Centring on zero also lets the
single-year noise band be drawn to scale, which turns Section 4.1's statement that
"single-year differences of 0.01 to 0.02 can never be conclusive" into something visible:
every point, winner or loser, sits inside it.

Confidence intervals are deliberately NOT drawn on the differences. The intervals in
Table 4.2 are on each condition's own IC, and two marginal intervals cannot be combined
into an interval on their difference, because the two models are scored on the same firms
and their errors are correlated. The paired Diebold-Mariano test is the honest statement
about a difference, and it is carried here by the marker fill.

NUMBERS ARE EMBEDDED LITERALS from Table 4.2 (tab:encgrid) and Table 4.1 (tab:anchor).
"""

import matplotlib.pyplot as plt

from _style import C, TEXTWIDTH_IN, apply_style, save

REF_IC = 0.603                 # struct+tfidf [sparse], Table 4.1
NOISE_HALFWIDTH = 0.06         # Section 4.1: the single-year CI spans roughly +/- 0.06
ALPHA = 0.05

# ---- Table 4.2, regrouped by what the encoder was trained to do -------------
# (label, IC, DM p)
FAMILIES = [
    ("General-purpose and similarity-trained", [
        ("enc[sbert] [ridge]",   0.562, 0.000),
        ("enc[sbert] [hgb]",     0.591, 0.043),
        ("enc[bge] [ridge]",     0.580, 0.000),
        ("enc[bge] [hgb]",       0.584, 0.023),
        ("enc[dual] [ridge]",    0.582, 0.013),
        ("enc[dual] [hgb]",      0.582, 0.051),
    ]),
    ("Task-aligned, trained on the volatility target", [
        ("enc[volaware] [ridge]",        0.587, 0.003),
        ("enc[volaware] [hgb]",          0.597, 0.593),
        ("enc[ftvol] [ridge]",           0.606, 0.207),
        ("enc[ftvol] [hgb]",             0.598, 0.155),
        ("enc[ftvol, topk_risk] [hgb]",  0.607, 0.454),
    ]),
    ("Dense-projected text and full fusion", [
        ("tfidf_svd [ridge]",                       0.594, 0.000),
        ("EVERYTHING svd+enc[ftvol]+chg [hgb]",     0.608, 0.204),
        ("EVERYTHING tfidf+enc[ftvol]+chg [sparse]", 0.590, 0.135),
    ]),
]


def main():
    apply_style()

    # lay out rows bottom-up, leaving a slot above each family for its header
    rows, headers, y = [], [], 0.0
    for title, members in reversed(FAMILIES):
        for label, ic, dm in reversed(members):
            rows.append((y, label, ic - REF_IC, dm))
            y += 1.0
        headers.append((y - 0.35, title))
        y += 1.1

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 6.0))

    ax.axvspan(-NOISE_HALFWIDTH, NOISE_HALFWIDTH, color=C["faint"], alpha=0.28,
               linewidth=0, zorder=0)
    ax.axvline(0.0, color=C["blue"], linestyle="--", linewidth=1.8, zorder=1)

    for ypos, label, delta, dm in rows:
        worse = dm < ALPHA
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
        ax.text(-NOISE_HALFWIDTH - 0.012, ypos, title, fontsize=11.5, style="italic",
                color="#333333", ha="left", va="center")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontfamily="monospace", fontsize=9.5)
    ax.set_ylim(-0.9, y - 0.5)
    ax.set_xlim(-NOISE_HALFWIDTH - 0.014, NOISE_HALFWIDTH + 0.014)
    ax.set_xlabel(r"IC difference from struct+tfidf [sparse]")
    ax.grid(axis="y", visible=False)

    xt = ax.get_xaxis_transform()
    ax.text(0.0, 1.015, "struct+tfidf [sparse]", transform=xt, color=C["blue"],
            fontsize=10.5, ha="center", va="bottom", fontfamily="monospace")
    ax.text(NOISE_HALFWIDTH - 0.002, 0.015, "single-year noise band", transform=xt,
            color="#555555", fontsize=10.5, ha="right", va="bottom")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=C["vermillion"],
                   label=rf"significantly worse, DM $p < {ALPHA:g}$"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor="white", markeredgecolor=C["blue"],
                   markeredgewidth=1.9, color=C["blue"],
                   label="tie: not separable from the reference"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=1)

    save(fig, "fig2_extra_delta_by_family", exact_width=False)


if __name__ == "__main__":
    main()

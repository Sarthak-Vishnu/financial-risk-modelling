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
count model, and it currently arrives as a fourteen-row table the reader has to scan
and hold in memory. A reference line turns that into one glance.

The intervals are the second reason. Section 4.1 states in prose that with n = 393 the
interval spans roughly +/- 0.06, which is why single-year differences of 0.01 to 0.02
can never be conclusive. Drawing the intervals shows it instead of asserting it: every
bar straddles the reference line, so the plot makes the case for moving to multi-year
paired tests before the reader reaches Section 4.3.

Rows are sorted by IC, best at the top, so the reading is top-down.

NUMBERS ARE EMBEDDED LITERALS, taken from Table 4.2 (tab:encgrid) of thesis.tex, with
the two reference lines from Table 4.1 (tab:anchor). They are not recomputed here.
R^2_log is deliberately not drawn: it is a third axis the plot has no room for, and it
survives in the appendix version of the table.
"""

import matplotlib.pyplot as plt

from _style import C, TEXTWIDTH_IN, apply_style, save

# ---- Table 4.2 (tab:encgrid): condition, IC, CI low, CI high, DM p ----------
GRID = [
    ("struct+enc[dual] [ridge]",                  0.582, 0.509, 0.652, 0.013),
    ("struct+enc[dual] [hgb]",                    0.582, 0.511, 0.651, 0.051),
    ("struct+enc[sbert] [ridge]",                 0.562, 0.482, 0.636, 0.000),
    ("struct+enc[sbert] [hgb]",                   0.591, 0.521, 0.658, 0.043),
    ("struct+enc[bge] [ridge]",                   0.580, 0.506, 0.649, 0.000),
    ("struct+enc[bge] [hgb]",                     0.584, 0.512, 0.651, 0.023),
    ("struct+enc[volaware] [ridge]",              0.587, 0.514, 0.656, 0.003),
    ("struct+enc[volaware] [hgb]",                0.597, 0.530, 0.663, 0.593),
    ("struct+enc[ftvol] [ridge]",                 0.606, 0.540, 0.672, 0.207),
    ("struct+enc[ftvol] [hgb]",                   0.598, 0.529, 0.666, 0.155),
    ("struct+enc[ftvol, topk_risk] [hgb]",        0.607, 0.538, 0.677, 0.454),
    ("struct+tfidf_svd [ridge]",                  0.594, 0.526, 0.661, 0.000),
    ("EVERYTHING svd+enc[ftvol]+chg [hgb]",       0.608, 0.542, 0.675, 0.204),
    ("EVERYTHING tfidf+enc[ftvol]+chg [sparse]",  0.590, 0.525, 0.662, 0.135),
]

# ---- Table 4.1 (tab:anchor): the two reference levels -----------------------
REF_DEFENDED = ("struct+tfidf [sparse]", 0.603)
REF_STRUCTURED = ("structured [ridge]", 0.591)

ALPHA = 0.05

# The 'struct+' prefix is on ten of fourteen rows and carries no information.
# Set False to print the condition names exactly as Table 4.2 does.
SHORT_LABELS = True

# Diagonal y-labels were tried at 25 degrees and made this figure worse, so the default
# is back to horizontal. The reason is geometry rather than taste: the longest label is
# about 3.2in of text, and rotating it by 25 degrees drops its far end 1.3in below its
# own tick, which is roughly four row heights at fourteen rows. Every label then visually
# spans four rows and the reader cannot tell which row it belongs to. Tilting pays only
# when labels are short or rows are tall, and here neither holds. Set to 25 to see it.
LABEL_ROTATION = 0


def label_for(name):
    if SHORT_LABELS and name.startswith("struct+"):
        return name[len("struct+"):]
    return name


def main():
    apply_style()
    rows = sorted(GRID, key=lambda r: r[1])          # ascending, so best ends on top
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 5.4))

    for i, (name, ic, lo, hi, dm) in zip(y, rows):
        worse = dm < ALPHA
        colour = C["vermillion"] if worse else C["blue"]
        ax.plot([lo, hi], [i, i], color=colour, linewidth=1.8, solid_capstyle="butt",
                zorder=2)
        ax.scatter([ic], [i], s=80, zorder=3, color=colour,
                   facecolors=colour if worse else "white",
                   edgecolors=colour, linewidths=1.8)

    ax.axvline(REF_DEFENDED[1], color=C["blue"], linestyle="--", linewidth=1.6,
               zorder=1)
    ax.axvline(REF_STRUCTURED[1], color=C["grey"], linestyle=":", linewidth=1.6,
               zorder=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels([label_for(r[0]) for r in rows],
                       fontfamily="monospace", fontsize=10,
                       rotation=LABEL_ROTATION, ha="right", va="center",
                       rotation_mode="anchor")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Information coefficient (2025 validation year)")
    ax.set_xlim(0.46, 0.70)
    ax.grid(axis="y", visible=False)

    # Reference lines named ABOVE the axes, at two heights so the two labels cannot
    # collide: the lines they mark are only 0.012 apart.
    xt = ax.get_xaxis_transform()
    ax.text(REF_STRUCTURED[1], 1.11, REF_STRUCTURED[0], transform=xt, color=C["grey"],
            fontsize=10.5, ha="center", va="bottom", fontfamily="monospace")
    ax.text(REF_DEFENDED[1], 1.02, REF_DEFENDED[0], transform=xt, color=C["blue"],
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
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.32), ncol=1)

    save(fig, "fig2_encoder_grid", exact_width=False)


if __name__ == "__main__":
    main()

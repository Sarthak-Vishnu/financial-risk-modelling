"""Shared style for the Chapter 4 result figures.

WIDTH IS MEASURED, NOT GUESSED. infthesis loads geometry with a4paper, left=4cm,
right=2.5cm, so \\textwidth is 21 - 4 - 2.5 = 14.5cm = 5.709in. Every figure is drawn
at exactly that width, which means \\includegraphics[width=\\textwidth] applies no
scaling and the point sizes set here are the point sizes that land on the page. Draw a
figure at some other width and scaling silently changes every font size in it.

Submission is digital, so colour carries meaning on its own and line style does not
have to duplicate it. The palette is Okabe-Ito anyway: it costs nothing and stays
legible under colour-vision deficiency.

NO FIGURE CARRIES A TITLE. The LaTeX caption does that job. A title inside the figure
duplicates the caption at a different size and in a different font.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # no display on a cluster or in a bare shell
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

TEXTWIDTH_IN = 14.5 / 2.54     # 5.709

# Okabe-Ito
C = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "orange":     "#E69F00",
    "pink":       "#CC79A7",
    "sky":        "#56B4E9",
    "grey":       "#666666",
    "faint":      "#BBBBBB",
}


def apply_style():
    """Font sizes are deliberately large. These figures are read on screen at
    \\textwidth in a 12pt document, and anything under 11pt disappears there."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,            # embed TrueType rather than Type 3
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    })


def save(fig, name, exact_width=True):
    """PDF for the thesis, PNG for eyeballing the result quickly.

    exact_width writes the file at exactly TEXTWIDTH_IN, so
    \\includegraphics[width=\\textwidth] applies no scaling and the point sizes set in
    apply_style() are the sizes that land on the page.

    This matters more than it sounds, and getting it wrong is invisible until the
    figures sit side by side in the document. bbox_inches="tight" crops to content, so
    three figures with different label widths come out at three different widths, and
    width=\\textwidth then stretches each by a different factor. Measured on the first
    version of these three: 390.6, 378.2 and 363.3 pt against a 411 pt text width, which
    is 5%, 9% and 13% of upscaling, so a 12pt label renders at 12.6, 13.1 and 13.6pt in
    the same chapter. tight_layout fits the content inside the fixed canvas instead of
    shrinking the canvas to the content.

    Pass exact_width=False for a figure with elements placed outside the axes, such as a
    legend anchored below them, which tight_layout does not account for and will clip."""
    OUT.mkdir(exist_ok=True)
    if exact_width:
        if fig.get_layout_engine() is None:
            fig.tight_layout(pad=0.35)
        kw = {}
    else:
        kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", **kw)

    # Fixing the canvas size means content can now run off it, which a cropped save
    # would silently have absorbed. Check rather than trust: a rotated multi-line
    # y-label is the usual culprit, because its text length becomes its vertical
    # extent and it can end up taller than the panel it labels.
    if exact_width:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        over = [side for side, bad in (("left", bb.x0 < -0.01), ("bottom", bb.y0 < -0.01),
                                       ("right", bb.x1 > w + 0.01), ("top", bb.y1 > h + 0.01))
                if bad]
        if over:
            print(f"  WARNING: {name} content overflows the canvas on the "
                  f"{', '.join(over)}. It will be clipped in the PDF.")

    print(f"wrote {OUT / (name + '.pdf')}  and  {OUT / (name + '.png')}")
    plt.close(fig)


def window_positions(windows):
    """Plot the measurement windows at evenly spaced positions, not at their values.

    Linear is unusable: 3, 5, 7 and 10 days would occupy the first eighth of the axis,
    and that is where all the structure sits. Log was the first alternative tried and is
    worse than it looks, because it encodes RATIO rather than elapsed time. Under log the
    step from 10 to 20 days is drawn wider than the step from 60 to 90, even though the
    first spans ten days and the second thirty. A reader taking the axis to mean duration
    is misled either way.

    Evenly spaced positions claim nothing about the gaps at all. These are eight chosen
    settings rather than samples of a continuum, every claim made about them concerns
    order and location, and nobody reads a slope off these panels. The tick labels carry
    the real values, so the axis is honest about what it is."""
    return list(range(len(windows)))


def window_axis(ax, windows):
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlim(-0.35, len(windows) - 0.65)

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
    return list(range(len(windows)))

def window_axis(ax, windows):
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlim(-0.35, len(windows) - 0.65)

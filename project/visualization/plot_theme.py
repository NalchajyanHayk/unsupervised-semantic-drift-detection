"""Publication-quality plotting theme for the drift-detection figures.

Exposes:
    apply_paper_theme()       set rcParams once for the whole module
    MODEL_COLORS              Okabe-Ito palette pinned to encoder names
    SIGNAL_COLORS             palette for the five drift signals
    NEUTRAL_GRAY              for threshold / baseline lines
    DRIFT_RED                 for drift markers / consensus bands
    FIG_SIZES                 inch tuples sized for a two-column LaTeX layout
    save_figure(fig, stem)    write .pdf + .png siblings (300 dpi)
    integer_year_axis(ax, …)  format an x-axis as integer years
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# --------------------------------------------------------------------------- #
# Palette                                                                      #
# --------------------------------------------------------------------------- #

# Okabe-Ito colorblind-safe palette.
_OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
}

MODEL_COLORS: dict[str, str] = {
    "MiniLM":     _OKABE_ITO["blue"],
    "LaBSE":      _OKABE_ITO["vermillion"],
    "DistilBERT": _OKABE_ITO["bluish_green"],
}

SIGNAL_COLORS: dict[str, str] = {
    "AE":  _OKABE_ITO["blue"],
    "JSD": _OKABE_ITO["orange"],
    "KS":  _OKABE_ITO["bluish_green"],
    "AD":  _OKABE_ITO["reddish_purple"],
    "SWD": _OKABE_ITO["vermillion"],
}

NEUTRAL_GRAY = "#555555"
LIGHT_GRAY   = "#BBBBBB"
DRIFT_RED    = "#C0392B"

# Sized for a typical two-column LaTeX page (textwidth ~ 6.9 in).
FIG_SIZES: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.6),
    "double": (7.0, 3.2),
    "tall":   (3.5, 4.0),
    "wide":   (7.0, 2.4),
}


# --------------------------------------------------------------------------- #
# Theme                                                                        #
# --------------------------------------------------------------------------- #

_THEME_APPLIED = False


def _preferred_sans_family() -> list[str]:
    """Return a sans-serif font stack with safe fallbacks."""
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    preferred = [
        "Inter",
        "Source Sans 3",
        "Source Sans Pro",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ]
    stack = [f for f in preferred if f in available]
    if "DejaVu Sans" not in stack:
        stack.append("DejaVu Sans")
    return stack


def apply_paper_theme(force: bool = False) -> None:
    """Apply paper-grade rcParams. Idempotent unless ``force=True``."""
    global _THEME_APPLIED
    if _THEME_APPLIED and not force:
        return

    family = _preferred_sans_family()

    mpl.rcParams.update({
        # Fonts ---------------------------------------------------------------
        "font.family":       "sans-serif",
        "font.sans-serif":   family,
        "font.size":         9.0,
        "axes.titlesize":    10.0,
        "axes.labelsize":    9.0,
        "xtick.labelsize":   8.0,
        "ytick.labelsize":   8.0,
        "legend.fontsize":   8.0,
        "figure.titlesize":  10.5,
        "mathtext.fontset":  "stix",

        # Lines / markers ----------------------------------------------------
        "lines.linewidth":   1.6,
        "lines.markersize":  4.0,
        "lines.markeredgewidth": 0.0,
        "patch.linewidth":   0.6,

        # Axes / spines ------------------------------------------------------
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#222222",
        "axes.titlepad":     6.0,
        "axes.titleweight":  "regular",
        "axes.axisbelow":    True,

        # Grid (y-only, light, behind data) -----------------------------------
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.color":        LIGHT_GRAY,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,

        # Ticks --------------------------------------------------------------
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        # Legend -------------------------------------------------------------
        "legend.frameon":         False,
        "legend.handlelength":    1.8,
        "legend.handletextpad":   0.5,
        "legend.columnspacing":   1.2,
        "legend.borderaxespad":   0.4,
        "legend.labelspacing":    0.35,

        # Figure / output ----------------------------------------------------
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches":0.02,
        "pdf.fonttype":      42,   # TrueType, so LaTeX can pick fonts
        "ps.fonttype":       42,
        "figure.autolayout": False,
    })

    _THEME_APPLIED = True


# --------------------------------------------------------------------------- #
# Saving                                                                       #
# --------------------------------------------------------------------------- #

def save_figure(fig, path_stem) -> tuple[Path, Path]:
    """Save ``fig`` as both .pdf (vector) and .png (300 dpi) siblings.

    ``path_stem`` may carry any common image suffix (``.png``, ``.pdf``,
    ``.jpg``); the suffix is stripped to derive the file stem so existing
    call sites that pass ``fig_path / 'foo.png'`` keep working unchanged.

    The legacy ``ae_reconstruction_error_post_gap 2.png`` artefact (a
    duplicate that older code accidentally produced) is never re-created
    because we always write to the exact derived stem.
    """
    p = Path(path_stem)
    suffix = p.suffix.lower()
    if suffix in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
        p = p.with_suffix("")

    p.parent.mkdir(parents=True, exist_ok=True)

    png_path = p.with_suffix(".png")
    pdf_path = p.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def integer_year_axis(ax, years: Iterable[int] | None = None,
                      nbins: int | None = None) -> None:
    """Force integer year ticks at a sensible density.

    ``nbins`` caps the number of major ticks shown -- pass a small number
    (e.g. 3-4) for narrow facets where labels would otherwise overlap.
    """
    if nbins is None:
        if years is not None:
            ys = [int(y) for y in years if y is not None]
            if ys:
                span = max(ys) - min(ys)
                nbins = 4 if span <= 6 else 6 if span <= 12 else 7
            else:
                nbins = 6
        else:
            nbins = 6
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True, prune=None, nbins=nbins,
                    steps=[1, 2, 5, 10])
    )
    ax.tick_params(axis="x", rotation=0)


def add_split_markers(ax, train_end: int | None, gap_end: int | None,
                      include_legend: bool = True) -> None:
    """Add dashed neutral lines for train_end and gap_end."""
    if train_end is not None:
        ax.axvline(train_end, linestyle=":", color=NEUTRAL_GRAY,
                   linewidth=0.9, alpha=0.9,
                   label="Train end" if include_legend else None)
    if gap_end is not None:
        ax.axvline(gap_end, linestyle=":", color=NEUTRAL_GRAY,
                   linewidth=0.9, alpha=0.9,
                   label="Gap end" if include_legend else None)


def place_legend(ax, ncol_below: int = 3) -> None:
    """Frameless legend; inside if it fits, otherwise below in one row."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if len(handles) <= 3:
        ax.legend(loc="best", frameon=False)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=ncol_below, frameon=False)


def color_for_model(name: str) -> str:
    """Look up a model color tolerantly (case-insensitive, slug-friendly)."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    key = name.replace("_", "").replace("-", "").lower()
    for k, v in MODEL_COLORS.items():
        if k.lower() == key:
            return v
    # Slug variants like "minilm", "labse", "distilbert"
    fallback = {"minilm": MODEL_COLORS["MiniLM"],
                "labse":  MODEL_COLORS["LaBSE"],
                "distilbert": MODEL_COLORS["DistilBERT"]}
    return fallback.get(key, _OKABE_ITO["black"])

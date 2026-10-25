"""Plot helpers: one consistent, quiet style for every figure + word clouds."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from . import paths

# categorical slots (fixed order, never cycled) and a one-hue sequential ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def apply_style():
    # the Ethiopic fallback font has no bold face; matplotlib would warn on every bold title
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if paths.FONT.exists():
        font_manager.fontManager.addfont(str(paths.FONT))
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Noto Sans Ethiopic"],   # Ethiopic glyphs fall back
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 7,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def save(fig, name: str):
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / name)


def label_line_ends(ax, xs, series: dict, fmt="{:.0f}"):
    """Value label at the end of each line (text in ink, never series colour)."""
    for name, ys in series.items():
        ax.annotate(fmt.format(ys[-1]), (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)


# --------------------------------------------------------------------------
# word clouds
# --------------------------------------------------------------------------
def _ramp_color_func(freqs: dict):
    lo, hi = np.log(min(freqs.values())), np.log(max(freqs.values()))

    def color(word, **_):
        t = 0.0 if hi == lo else (np.log(freqs[word]) - lo) / (hi - lo)
        return BLUE_RAMP[int(round(t * (len(BLUE_RAMP) - 1)))]
    return color


def wordcloud(freqs: dict, ax=None, title: str = "", max_words: int = 150):
    """Render a word cloud from {phrase: frequency} with an Ethiopic font."""
    from wordcloud import WordCloud
    wc = WordCloud(font_path=str(paths.FONT), width=900, height=520, background_color=SURFACE,
                   max_words=max_words, prefer_horizontal=1.0, relative_scaling=0.45,
                   collocations=False, margin=4, random_state=7)
    wc.generate_from_frequencies(freqs)
    wc.recolor(color_func=_ramp_color_func(freqs), random_state=7)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax

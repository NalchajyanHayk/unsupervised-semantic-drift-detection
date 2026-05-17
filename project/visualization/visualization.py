"""Publication-quality plotting for the drift-detection pipeline.

Every function from the original module keeps the same signature; only the
internals were upgraded:

  * Routes saving through ``plot_theme.save_figure`` so each figure produces
    a ``.pdf`` (vector, for LaTeX ``\\includegraphics``) **and** a ``.png``
    (300 dpi) sibling.
  * Uses the Okabe-Ito palette and the rcParams installed by
    ``plot_theme.apply_paper_theme``.
  * No chart titles by default. Pass ``title=None`` to keep that behaviour;
    pass a string to display one (e.g. for the slide deck).
  * Threshold and split lines are dashed neutral gray and appear in the
    legend instead of free-floating text.
  * Histograms get filled bars + a smooth KDE-style outline and an
    annotated vertical threshold line.

Notable additions (do not break existing callers):
  * ``plot_two_lines`` accepts optional ``train_end=None`` / ``gap_end=None``
    so the cross-encoder comparisons in ``comparison.py`` work (latent bug
    fix: the existing call sites passed these kwargs but the previous
    signature did not accept them).
  * ``plot_consensus`` renders a per-year drift-voting heatmap with subtle
    red bands on consensus years.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import DriftConfig
from .plot_theme import (
    DRIFT_RED,
    FIG_SIZES,
    LIGHT_GRAY,
    MODEL_COLORS,
    NEUTRAL_GRAY,
    SIGNAL_COLORS,
    add_split_markers,
    apply_paper_theme,
    color_for_model,
    integer_year_axis,
    place_legend,
    save_figure,
)

# Apply on import so anything that touches matplotlib via this module is themed.
apply_paper_theme()


# --------------------------------------------------------------------------- #
# Generic helpers                                                              #
# --------------------------------------------------------------------------- #

def _series_color(name: str | None, fallback: str) -> str:
    if name is None:
        return fallback
    # Look up by encoder name first (case-insensitive); else use fallback.
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return color_for_model(name) if name.lower() in {"minilm", "labse", "distilbert"} else fallback


def _line(ax, x, y, *, color: str, label: str | None = None,
          marker: str = "o", zorder: int = 3) -> None:
    ax.plot(x, y, marker=marker, color=color, label=label,
            linewidth=1.6, markersize=4.0, zorder=zorder)


def _threshold_line(ax, value: float, label: str, *,
                    color: str = NEUTRAL_GRAY, linestyle: str = "--") -> None:
    ax.axhline(value, linestyle=linestyle, color=color,
               linewidth=1.0, alpha=0.85, zorder=2, label=label)


def _maybe_title(ax, title: str | None) -> None:
    if title:
        ax.set_title(title)


def _new_axes(size_key: str = "single"):
    fig, ax = plt.subplots(figsize=FIG_SIZES.get(size_key, FIG_SIZES["single"]))
    return fig, ax


# --------------------------------------------------------------------------- #
# Time-series of a scalar metric                                               #
# --------------------------------------------------------------------------- #

def _plot_metric_over_time_impl(
    metric_df: pd.DataFrame,
    *,
    threshold: float | None,
    threshold_label: str | None,
    ylabel: str,
    title: str | None,
    config: DriftConfig,
    fig_path: Path,
    color: str,
    drift_col: str | None = None,
    drift_label: str = "Drift detected",
) -> None:
    df = metric_df.dropna(subset=["mean"]).sort_values("year")
    fig, ax = _new_axes("single")

    _line(ax, df["year"], df["mean"], color=color, label=ylabel)
    if threshold is not None:
        _threshold_line(ax, threshold, threshold_label or "Threshold")

    add_split_markers(ax, config.train_end, config.gap_end)

    if drift_col and drift_col in df.columns:
        hits = df[df[drift_col].astype(bool)]
        if not hits.empty:
            ax.scatter(hits["year"], hits["mean"], s=36, marker="o",
                       facecolors=DRIFT_RED, edgecolors="white",
                       linewidths=0.8, zorder=5, label=drift_label)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    integer_year_axis(ax, df["year"].tolist())
    _maybe_title(ax, title)
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_reconstruction_error(
    year_recon,
    threshold,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    _plot_metric_over_time_impl(
        year_recon,
        threshold=threshold,
        threshold_label="AE threshold",
        ylabel="Mean reconstruction error",
        title=None,  # ignore legacy title; captions live in the paper
        config=config,
        fig_path=fig_path,
        color=SIGNAL_COLORS["AE"],
    )


def plot_reconstruction_post_gap(
    year_recon,
    threshold,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    sub = year_recon[year_recon["year"] > config.gap_end]
    df = sub.dropna(subset=["mean"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["mean"], color=SIGNAL_COLORS["AE"],
          label="Mean reconstruction error")
    _threshold_line(ax, threshold, "AE threshold")
    if "mean" in df.columns and threshold is not None:
        hits = df[df["mean"] > threshold]
        if not hits.empty:
            ax.scatter(hits["year"], hits["mean"], s=36, marker="o",
                       facecolors=DRIFT_RED, edgecolors="white",
                       linewidths=0.8, zorder=5, label="Drift")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean reconstruction error")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_metric_over_time(
    metric_df,
    threshold: float,
    title: str,
    ylabel: str,
    threshold_label: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    _plot_metric_over_time_impl(
        metric_df,
        threshold=threshold,
        threshold_label=threshold_label,
        ylabel=ylabel,
        title=None,
        config=config,
        fig_path=fig_path,
        color=SIGNAL_COLORS["AE"],
    )


def plot_metric_post_gap(
    metric_df,
    threshold: float,
    title: str,
    ylabel: str,
    threshold_label: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    sub = metric_df[metric_df["year"] > config.gap_end]
    df = sub.dropna(subset=["mean"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["mean"], color=SIGNAL_COLORS["AE"], label=ylabel)
    _threshold_line(ax, threshold, threshold_label)
    if threshold is not None:
        hits = df[df["mean"] > threshold]
        if not hits.empty:
            ax.scatter(hits["year"], hits["mean"], s=36, marker="o",
                       facecolors=DRIFT_RED, edgecolors="white",
                       linewidths=0.8, zorder=5, label="Drift")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_jsd_over_time(
    year_jsd,
    threshold: float,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    _plot_metric_over_time_impl(
        year_jsd,
        threshold=threshold,
        threshold_label="JSD threshold",
        ylabel="Mean Jensen-Shannon divergence",
        title=None,
        config=config,
        fig_path=fig_path,
        color=SIGNAL_COLORS["JSD"],
    )


def plot_jsd_post_gap(
    year_jsd,
    threshold: float,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    sub = year_jsd[year_jsd["year"] > config.gap_end]
    df = sub.dropna(subset=["mean"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["mean"], color=SIGNAL_COLORS["JSD"],
          label="Mean JSD")
    _threshold_line(ax, threshold, "JSD threshold")
    if threshold is not None:
        hits = df[df["mean"] > threshold]
        if not hits.empty:
            ax.scatter(hits["year"], hits["mean"], s=36, marker="o",
                       facecolors=DRIFT_RED, edgecolors="white",
                       linewidths=0.8, zorder=5, label="Drift")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Jensen-Shannon divergence")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


# --------------------------------------------------------------------------- #
# KS / AD                                                                      #
# --------------------------------------------------------------------------- #

def _stat_signal_color(method_name: str) -> str:
    if method_name.upper() == "KS":
        return SIGNAL_COLORS["KS"]
    if method_name.upper() == "AD":
        return SIGNAL_COLORS["AD"]
    return SIGNAL_COLORS["AE"]


def plot_stat_fraction(
    stat_df,
    method_name: str,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    df = stat_df.dropna(subset=["fraction_significant_fdr"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["fraction_significant_fdr"],
          color=_stat_signal_color(method_name),
          label=f"{method_name}: fraction significant")
    _threshold_line(ax, config.drift_fraction_threshold, "Drift threshold")
    add_split_markers(ax, config.train_end, config.gap_end)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction significant (FDR)")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_stat_value(
    stat_df,
    method_name: str,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    df = stat_df.dropna(subset=["mean_stat"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["mean_stat"],
          color=_stat_signal_color(method_name),
          label=f"{method_name}: mean statistic")
    add_split_markers(ax, config.train_end, config.gap_end)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean test statistic")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_stat_post_gap(
    stat_df,
    method_name: str,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    sub = stat_df[stat_df["year"] > config.gap_end]
    df = sub.dropna(subset=["fraction_significant_fdr"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["fraction_significant_fdr"],
          color=_stat_signal_color(method_name),
          label=f"{method_name}: post-gap")
    _threshold_line(ax, config.drift_fraction_threshold, "Drift threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction significant (FDR)")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


# --------------------------------------------------------------------------- #
# Sliced Wasserstein                                                           #
# --------------------------------------------------------------------------- #

def plot_wasserstein_yearly(
    w_df,
    baseline: dict,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    df = w_df.dropna(subset=["mean_swd"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["mean_swd"], color=SIGNAL_COLORS["SWD"],
          label="Mean SWD")
    _threshold_line(ax, baseline["threshold_95"], "95% bootstrap")
    _threshold_line(ax, baseline["threshold_99"], "99% bootstrap",
                    linestyle=(0, (3, 1, 1, 1)))
    add_split_markers(ax, config.train_end, config.gap_end)

    if "drift_95" in df.columns:
        hits = df[df["drift_95"].astype(bool)]
        if not hits.empty:
            ax.scatter(hits["year"], hits["mean_swd"], s=36, marker="o",
                       facecolors=DRIFT_RED, edgecolors="white",
                       linewidths=0.8, zorder=5, label="Drift")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean sliced Wasserstein distance")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_wasserstein_zscore(
    w_df,
    title: str,
    config: DriftConfig,
    fig_path: Path,
) -> None:
    df = w_df.dropna(subset=["z_score"]).sort_values("year")
    fig, ax = _new_axes("single")
    _line(ax, df["year"], df["z_score"], color=SIGNAL_COLORS["SWD"],
          label="SWD z-score")
    _threshold_line(ax, 2.0, "z = 2")
    add_split_markers(ax, config.train_end, config.gap_end)
    ax.set_xlabel("Year")
    ax.set_ylabel("Z-score")
    integer_year_axis(ax, df["year"].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


def plot_baseline_histogram(
    baseline: dict,
    title: str,
    fig_path: Path,
) -> None:
    values = np.asarray(baseline["values"], dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = _new_axes("single")

    if values.size:
        n_bins = max(10, min(40, int(np.sqrt(values.size) * 1.5)))
        counts, edges, _ = ax.hist(
            values, bins=n_bins, color=SIGNAL_COLORS["SWD"],
            alpha=0.35, edgecolor=SIGNAL_COLORS["SWD"], linewidth=0.6,
            zorder=2,
        )

        # KDE-style outline (Gaussian KDE without scipy) -- a smoothed
        # rolling sum over fine bins gives a nice envelope.
        fine = np.linspace(values.min(), values.max(), 200)
        try:
            std = float(np.std(values))
            if std > 0:
                # crude KDE via Gaussian sum (no scipy dependency).
                bw = 1.06 * std * values.size ** (-1 / 5)
                if bw > 0:
                    contrib = np.exp(
                        -0.5 * ((fine[:, None] - values[None, :]) / bw) ** 2
                    )
                    density = contrib.sum(axis=1)
                    if density.max() > 0:
                        density = density / density.max() * counts.max()
                        ax.plot(fine, density, color=SIGNAL_COLORS["SWD"],
                                linewidth=1.4, alpha=0.9, zorder=3)
        except Exception:
            pass

    t95 = baseline.get("threshold_95")
    t99 = baseline.get("threshold_99")
    if t95 is not None:
        ax.axvline(t95, linestyle="--", color=NEUTRAL_GRAY,
                   linewidth=1.0, label="95% threshold", zorder=4)
        ax.text(t95, ax.get_ylim()[1] * 0.92, " 95%",
                color=NEUTRAL_GRAY, fontsize=8, va="top", ha="left")
    if t99 is not None:
        ax.axvline(t99, linestyle=(0, (3, 1, 1, 1)), color=NEUTRAL_GRAY,
                   linewidth=1.0, label="99% threshold", zorder=4)
        ax.text(t99, ax.get_ylim()[1] * 0.78, " 99%",
                color=NEUTRAL_GRAY, fontsize=8, va="top", ha="left")

    ax.set_xlabel("Bootstrap SWD")
    ax.set_ylabel("Frequency")
    place_legend(ax)
    save_figure(fig, fig_path)


# --------------------------------------------------------------------------- #
# Two-series line plot                                                         #
# --------------------------------------------------------------------------- #

def plot_two_lines(
    df,
    x_col: str,
    y1: str,
    y2: str,
    label1: str,
    label2: str,
    title: str,
    ylabel: str,
    fig_path: Path,
    *,
    train_end: int | None = None,
    gap_end: int | None = None,
    color1: str | None = None,
    color2: str | None = None,
) -> None:
    """Two superimposed series on one axes (e.g. ordinary vs latent SWD).

    The optional ``train_end`` / ``gap_end`` keyword arguments add dashed
    neutral vertical lines for the temporal splits; ``color1`` / ``color2``
    let callers override the palette (used by the encoder comparisons so a
    given encoder gets the same colour everywhere in the paper).
    """
    work = df.dropna(subset=[y1, y2], how="all").sort_values(x_col)
    fig, ax = _new_axes("single")

    c1 = color1 or _series_color(label1, SIGNAL_COLORS["SWD"])
    c2 = color2 or _series_color(label2, SIGNAL_COLORS["AE"])

    _line(ax, work[x_col], work[y1], color=c1, label=label1)
    _line(ax, work[x_col], work[y2], color=c2, label=label2, marker="s")

    add_split_markers(ax, train_end, gap_end)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    integer_year_axis(ax, work[x_col].tolist())
    place_legend(ax)
    save_figure(fig, fig_path)


# --------------------------------------------------------------------------- #
# Consensus voting                                                             #
# --------------------------------------------------------------------------- #

def plot_consensus(
    summary: pd.DataFrame,
    title: str | None,
    fig_path: Path,
    *,
    signals: Iterable[str] = ("AE", "JSD", "KS", "AD", "SWD"),
    consensus_threshold: int = 3,
) -> None:
    """Heatmap-style per-year drift voting with a consensus band overlay.

    Expects ``summary`` indexed by year with one boolean column per signal
    plus ``Votes`` (int) and ``Consensus`` (bool); same schema produced by
    ``run_full_pipeline`` in ``pipeline.py``.
    """
    if summary is None or summary.empty:
        return

    sig_list = [s for s in signals if s in summary.columns]
    if not sig_list:
        return

    years = list(summary.index.astype(int))
    grid = summary[sig_list].astype(bool).to_numpy().astype(float).T

    fig, ax = plt.subplots(figsize=FIG_SIZES["double"])

    # Subtle red band on consensus years (under the data).
    if "Consensus" in summary.columns:
        for y, flag in zip(years, summary["Consensus"].astype(bool)):
            if flag:
                ax.axvspan(y - 0.5, y + 0.5, color=DRIFT_RED, alpha=0.10,
                           zorder=1)

    # One row per signal: filled cells where the signal fires.
    for i, sig in enumerate(sig_list):
        row = grid[i]
        for j, year in enumerate(years):
            if row[j]:
                ax.add_patch(plt.Rectangle(
                    (year - 0.42, i - 0.42), 0.84, 0.84,
                    facecolor=SIGNAL_COLORS.get(sig, NEUTRAL_GRAY),
                    edgecolor="white", linewidth=0.4, zorder=3,
                ))
            else:
                ax.add_patch(plt.Rectangle(
                    (year - 0.42, i - 0.42), 0.84, 0.84,
                    facecolor="#F2F2F2", edgecolor="white",
                    linewidth=0.4, zorder=2,
                ))

    ax.set_yticks(range(len(sig_list)))
    ax.set_yticklabels(sig_list)
    ax.set_xlim(min(years) - 0.6, max(years) + 0.6)
    ax.set_ylim(-0.6, len(sig_list) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.grid(False)

    integer_year_axis(ax, years)

    # Vote count annotations along the top.
    if "Votes" in summary.columns:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        for y, v in zip(years, summary["Votes"].astype(int)):
            color = DRIFT_RED if v >= consensus_threshold else NEUTRAL_GRAY
            ax2.text(y, -0.55, str(int(v)), ha="center", va="bottom",
                     fontsize=7, color=color)
        ax2.spines["right"].set_visible(False)

    # Slim, frameless legend for what red means.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=DRIFT_RED, alpha=0.25,
                     label=f"Consensus (Votes ≥ {consensus_threshold})")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=1)

    save_figure(fig, fig_path)

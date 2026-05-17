"""Cross-encoder comparison figures and tables.

Same drift metrics, paper-grade styling:
  * encoders share the same colour everywhere (Okabe-Ito via plot_theme).
  * pairwise comparisons still emit the legacy ``..._comparison.png/.pdf``
    files (so existing references in the paper keep working), and also a
    new combined ``{a}_vs_{b}_panel.{pdf,png}`` small-multiple figure that
    shares the y-axis where it helps readability.
  * the all-encoder faceted view uses the same palette and saves both PDF
    and PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from visualization.plot_theme import (
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
    save_figure,
)
from .helpers import ensure_dir, slugify
from visualization.table_export import save_table
from visualization.visualization import (
    plot_baseline_histogram,
    plot_consensus,
    plot_jsd_over_time,
    plot_jsd_post_gap,
    plot_reconstruction_error,
    plot_reconstruction_post_gap,
    plot_stat_fraction,
    plot_stat_post_gap,
    plot_stat_value,
    plot_two_lines,
    plot_wasserstein_yearly,
    plot_wasserstein_zscore,
)
import numpy as np

apply_paper_theme()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _get_split_bounds(result: Dict[str, Any]) -> tuple[int | None, int | None]:
    return result.get("train_end"), result.get("gap_end")


def _safe_metric_df(result: Dict[str, Any], metric_key: str) -> pd.DataFrame:
    """Return ``year, value`` for the requested metric or an empty frame."""
    if metric_key == "ae" and "ae_yearly" in result:
        return result["ae_yearly"][["year", "mean"]].rename(columns={"mean": "value"})
    if metric_key == "jsd" and "jsd_yearly" in result:
        return result["jsd_yearly"][["year", "mean"]].rename(columns={"mean": "value"})
    if metric_key == "ordinary_swd" and "w_df" in result:
        return result["w_df"][["year", "mean_swd"]].rename(columns={"mean_swd": "value"})
    if metric_key == "latent_swd" and "latent_w_df" in result:
        return result["latent_w_df"][["year", "mean_swd"]].rename(
            columns={"mean_swd": "value"}
        )
    return pd.DataFrame(columns=["year", "value"])


def _encoder_color(name: str) -> str:
    return color_for_model(name)


# --------------------------------------------------------------------------- #
# Pairwise comparisons                                                         #
# --------------------------------------------------------------------------- #

def compare_two_embeddings(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    output_dir: Path,
) -> dict:
    name_a = slugify(results_a["name"])
    name_b = slugify(results_b["name"])
    pretty_a = results_a["name"]
    pretty_b = results_b["name"]
    color_a = _encoder_color(pretty_a)
    color_b = _encoder_color(pretty_b)

    fig_dir = ensure_dir(output_dir / "figures" / "comparisons")
    table_dir = ensure_dir(output_dir / "tables" / "comparisons")
    train_end, gap_end = _get_split_bounds(results_a)

    # Build the four merged dataframes (unchanged schemas).
    ae_a = results_a["ae_yearly"][["year", "mean", "drift_flag"]].rename(
        columns={"mean": f"{name_a}_ae_error", "drift_flag": f"{name_a}_ae_drift"}
    )
    ae_b = results_b["ae_yearly"][["year", "mean", "drift_flag"]].rename(
        columns={"mean": f"{name_b}_ae_error", "drift_flag": f"{name_b}_ae_drift"}
    )
    ae_cmp = ae_a.merge(ae_b, on="year", how="outer")

    jsd_a = results_a["jsd_yearly"][["year", "mean", "drift_flag"]].rename(
        columns={"mean": f"{name_a}_jsd", "drift_flag": f"{name_a}_jsd_drift"}
    )
    jsd_b = results_b["jsd_yearly"][["year", "mean", "drift_flag"]].rename(
        columns={"mean": f"{name_b}_jsd", "drift_flag": f"{name_b}_jsd_drift"}
    )
    jsd_cmp = jsd_a.merge(jsd_b, on="year", how="outer")

    swd_a = results_a["w_df"][["year", "mean_swd", "drift_95"]].rename(
        columns={"mean_swd": f"{name_a}_ordinary_swd",
                 "drift_95": f"{name_a}_ordinary_drift"}
    )
    swd_b = results_b["w_df"][["year", "mean_swd", "drift_95"]].rename(
        columns={"mean_swd": f"{name_b}_ordinary_swd",
                 "drift_95": f"{name_b}_ordinary_drift"}
    )
    swd_cmp = swd_a.merge(swd_b, on="year", how="outer")

    latent_a = results_a["latent_w_df"][["year", "mean_swd", "drift_95"]].rename(
        columns={"mean_swd": f"{name_a}_latent_swd",
                 "drift_95": f"{name_a}_latent_drift"}
    )
    latent_b = results_b["latent_w_df"][["year", "mean_swd", "drift_95"]].rename(
        columns={"mean_swd": f"{name_b}_latent_swd",
                 "drift_95": f"{name_b}_latent_drift"}
    )
    latent_cmp = latent_a.merge(latent_b, on="year", how="outer")

    # Tables (CSV + .tex sibling).
    save_table(
        ae_cmp, table_dir / f"{name_a}_vs_{name_b}_ae_comparison.csv",
        caption=f"AE reconstruction error: {pretty_a} vs {pretty_b}",
        label=f"tab:{name_a}_vs_{name_b}_ae",
    )
    save_table(
        jsd_cmp, table_dir / f"{name_a}_vs_{name_b}_jsd_comparison.csv",
        caption=f"Jensen-Shannon divergence: {pretty_a} vs {pretty_b}",
        label=f"tab:{name_a}_vs_{name_b}_jsd",
    )
    save_table(
        swd_cmp, table_dir / f"{name_a}_vs_{name_b}_ordinary_swd_comparison.csv",
        caption=f"Ordinary SWD: {pretty_a} vs {pretty_b}",
        label=f"tab:{name_a}_vs_{name_b}_ordinary_swd",
    )
    save_table(
        latent_cmp, table_dir / f"{name_a}_vs_{name_b}_latent_swd_comparison.csv",
        caption=f"AE-latent SWD: {pretty_a} vs {pretty_b}",
        label=f"tab:{name_a}_vs_{name_b}_latent_swd",
    )

    # Legacy single-axes figures (kept for back-compat with paper references).
    plot_two_lines(
        ae_cmp, "year",
        f"{name_a}_ae_error", f"{name_b}_ae_error",
        pretty_a, pretty_b,
        "AE reconstruction error comparison",
        "Mean reconstruction error",
        fig_dir / f"{name_a}_vs_{name_b}_ae_comparison.png",
        train_end=train_end, gap_end=gap_end,
        color1=color_a, color2=color_b,
    )
    plot_two_lines(
        jsd_cmp, "year",
        f"{name_a}_jsd", f"{name_b}_jsd",
        pretty_a, pretty_b,
        "Jensen-Shannon divergence comparison",
        "Mean JSD",
        fig_dir / f"{name_a}_vs_{name_b}_jsd_comparison.png",
        train_end=train_end, gap_end=gap_end,
        color1=color_a, color2=color_b,
    )
    plot_two_lines(
        swd_cmp, "year",
        f"{name_a}_ordinary_swd", f"{name_b}_ordinary_swd",
        pretty_a, pretty_b,
        "Ordinary SWD comparison",
        "Mean SWD",
        fig_dir / f"{name_a}_vs_{name_b}_ordinary_swd_comparison.png",
        train_end=train_end, gap_end=gap_end,
        color1=color_a, color2=color_b,
    )
    plot_two_lines(
        latent_cmp, "year",
        f"{name_a}_latent_swd", f"{name_b}_latent_swd",
        pretty_a, pretty_b,
        "AE-latent SWD comparison",
        "Mean latent SWD",
        fig_dir / f"{name_a}_vs_{name_b}_latent_swd_comparison.png",
        train_end=train_end, gap_end=gap_end,
        color1=color_a, color2=color_b,
    )

    # New: small-multiples panel (2x2) for nicer paper inclusion.
    _plot_pairwise_panel(
        pretty_a, pretty_b, color_a, color_b,
        ae_cmp, jsd_cmp, swd_cmp, latent_cmp,
        name_a, name_b,
        train_end, gap_end,
        fig_dir / f"{name_a}_vs_{name_b}_panel.png",
    )

    return {
        "ae_cmp": ae_cmp,
        "jsd_cmp": jsd_cmp,
        "swd_cmp": swd_cmp,
        "latent_cmp": latent_cmp,
    }


def _plot_pairwise_panel(
    pretty_a: str, pretty_b: str, color_a: str, color_b: str,
    ae_cmp: pd.DataFrame, jsd_cmp: pd.DataFrame,
    swd_cmp: pd.DataFrame, latent_cmp: pd.DataFrame,
    name_a: str, name_b: str,
    train_end: int | None, gap_end: int | None,
    fig_path: Path,
) -> None:
    panels = [
        ("AE reconstruction error", ae_cmp,
         f"{name_a}_ae_error", f"{name_b}_ae_error", "Mean error"),
        ("Jensen-Shannon divergence", jsd_cmp,
         f"{name_a}_jsd", f"{name_b}_jsd", "Mean JSD"),
        ("Ordinary SWD", swd_cmp,
         f"{name_a}_ordinary_swd", f"{name_b}_ordinary_swd", "Mean SWD"),
        ("AE-latent SWD", latent_cmp,
         f"{name_a}_latent_swd", f"{name_b}_latent_swd", "Mean SWD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=False)
    flat = axes.flatten()
    all_years: list[int] = []

    for ax, (title, df, col_a, col_b, ylabel) in zip(flat, panels):
        if df is None or df.empty or col_a not in df.columns or col_b not in df.columns:
            ax.set_visible(False)
            continue
        work = df.sort_values("year")
        ax.plot(work["year"], work[col_a], marker="o", color=color_a,
                linewidth=1.6, markersize=3.6, label=pretty_a)
        ax.plot(work["year"], work[col_b], marker="s", color=color_b,
                linewidth=1.6, markersize=3.6, label=pretty_b)
        add_split_markers(ax, train_end, gap_end, include_legend=False)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        all_years.extend(work["year"].dropna().astype(int).tolist())
        integer_year_axis(
            ax, work["year"].dropna().astype(int).tolist(), nbins=5,
        )

    # Single shared legend along the bottom (encoder identity only).
    handles = [
        plt.Line2D([0], [0], marker="o", color=color_a, linewidth=1.6,
                   markersize=4, label=pretty_a),
        plt.Line2D([0], [0], marker="s", color=color_b, linewidth=1.6,
                   markersize=4, label=pretty_b),
        plt.Line2D([0], [0], linestyle=":", color=NEUTRAL_GRAY,
                   linewidth=0.9, label="Train / gap split"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, fig_path)


# --------------------------------------------------------------------------- #
# All-encoder faceted comparison                                               #
# --------------------------------------------------------------------------- #

def compare_all_embeddings(
    results: list[Dict[str, Any]],
    output_dir: Path,
) -> None:
    if len(results) < 2:
        return

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            compare_two_embeddings(results[i], results[j], output_dir)

    plot_faceted_encoder_comparison(results, output_dir)


def plot_faceted_encoder_comparison(
    results: list[Dict[str, Any]],
    output_dir: Path,
    fig_name: str = "encoders_faceted_comparison.png",
) -> None:
    if not results:
        return

    fig_dir = ensure_dir(output_dir / "figures" / "comparisons")

    metric_specs = [
        ("ae",            "AE reconstruction error", "Mean error"),
        ("jsd",           "Jensen-Shannon divergence", "Mean JSD"),
        ("ordinary_swd",  "Ordinary SWD", "Mean SWD"),
        ("latent_swd",    "AE-latent SWD", "Mean SWD"),
    ]

    # Drop metrics that aren't present in any result (e.g. fast-path has no JSD).
    metric_specs = [
        spec for spec in metric_specs
        if any(not _safe_metric_df(r, spec[0]).empty for r in results)
    ]
    if not metric_specs:
        return

    n_rows = len(results)
    n_cols = len(metric_specs)

    # Per-metric y-limits shared across encoder rows for fair visual comparison.
    metric_limits: dict[str, tuple[float, float]] = {}
    for metric_key, _, _ in metric_specs:
        all_vals: list[float] = []
        for result in results:
            df = _safe_metric_df(result, metric_key)
            if not df.empty:
                all_vals.extend(df["value"].dropna().tolist())
        if not all_vals:
            metric_limits[metric_key] = (0.0, 1.0)
            continue
        y_min, y_max = float(min(all_vals)), float(max(all_vals))
        if y_min == y_max:
            pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
            metric_limits[metric_key] = (y_min - pad, y_max + pad)
        else:
            pad = (y_max - y_min) * 0.07
            metric_limits[metric_key] = (y_min - pad, y_max + pad)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 0.8, 1.7 * n_rows + 0.6),
        squeeze=False, sharex=False,
    )

    all_year_values: list[int] = []

    for row_idx, result in enumerate(results):
        encoder_name = result["name"]
        train_end, gap_end = _get_split_bounds(result)
        color = _encoder_color(encoder_name)

        for col_idx, (metric_key, metric_title, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            df = _safe_metric_df(result, metric_key).sort_values("year")

            if df.empty:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color=LIGHT_GRAY)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.plot(df["year"], df["value"], marker="o", color=color,
                        linewidth=1.6, markersize=3.6)
                add_split_markers(ax, train_end, gap_end, include_legend=False)
                ax.set_ylim(metric_limits[metric_key])
                integer_year_axis(
                    ax, df["year"].dropna().astype(int).tolist(), nbins=4,
                )
                all_year_values.extend(df["year"].dropna().astype(int).tolist())

            if row_idx == 0:
                ax.set_title(metric_title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(encoder_name, fontsize=9, color=color)
            else:
                ax.set_ylabel("")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Year")
            else:
                ax.set_xlabel("")

    # Legend describing the split markers + encoder color coding.
    handles = [
        plt.Line2D([0], [0], color=_encoder_color(r["name"]),
                   marker="o", linewidth=1.6, markersize=4,
                   label=r["name"])
        for r in results
    ]
    handles.append(
        plt.Line2D([0], [0], linestyle=":", color=NEUTRAL_GRAY,
                   linewidth=0.9, label="Train / gap split")
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, fig_dir / fig_name)


# --------------------------------------------------------------------------- #
# Tables -> figures: loader and regenerator                                    #
# --------------------------------------------------------------------------- #

def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _reconstruct_baseline(values_csv: Path) -> dict | None:
    """Build a baseline dict (values, threshold_95, threshold_99) from the
    saved bootstrap CSV. Returns None if the file is missing or empty."""
    if not values_csv.exists():
        return None
    df = pd.read_csv(values_csv)
    if df.empty:
        return None
    col = df.columns[0]
    values = df[col].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return {
        "values": values,
        "threshold_95": float(np.percentile(values, 95)),
        "threshold_99": float(np.percentile(values, 99)),
    }


def _threshold_from_period_summary(
    period_df: pd.DataFrame | None,
    mean_col: str,
    std_col: str,
    std_multiplier: float,
) -> float | None:
    """Train threshold = train_mean + multiplier * train_std (matches
    ``models.compute_threshold`` on the training subset)."""
    if period_df is None or period_df.empty:
        return None
    if "Period" not in period_df.columns:
        return None
    mask = period_df["Period"].astype(str).str.startswith("Train")
    if not mask.any():
        return None
    row = period_df.loc[mask].iloc[0]
    if mean_col not in row.index or std_col not in row.index:
        return None
    mean_v = row[mean_col]
    std_v = row[std_col]
    if pd.isna(mean_v) or pd.isna(std_v):
        return None
    return float(mean_v) + float(std_multiplier) * float(std_v)


def load_results_from_tables(
    names: list[str],
    output_dir: Path,
    train_end: int = 2005,
    gap_end: int = 2010,
    std_multiplier: float = 0.5,
) -> list[Dict[str, Any]]:
    """Load *every* per-encoder table from disk and reconstruct enough
    metadata (thresholds, baselines) to regenerate all figures.

    Backwards-compatible: still returns ``name, slug, train_end, gap_end,
    ae_yearly, jsd_yearly, w_df, latent_w_df`` keys (existing callers keep
    working); adds extra keys (``ks_df``, ``ad_df``, ``ae_threshold``,
    ``jsd_threshold``, ``baseline``, ``latent_baseline``, ``year_recon``,
    ``year_jsd``, ``swd_compare``, ``summary``, ``period_error_summary``,
    ``period_jsd_summary``, ``period_swd``, ``losses_df``).
    """
    results: list[Dict[str, Any]] = []
    table_root = output_dir / "tables"

    for name in names:
        slug = slugify(name)
        base = table_root / slug

        ae_all = _read_optional(base / f"{slug}_ae_all_years.csv")
        ae_yearly = _read_optional(base / f"{slug}_ae_yearly.csv")
        jsd_all = _read_optional(base / f"{slug}_jsd_all_years.csv")
        jsd_yearly = _read_optional(base / f"{slug}_jsd_yearly.csv")
        ks_df = _read_optional(base / f"{slug}_ks.csv")
        ad_df = _read_optional(base / f"{slug}_ad.csv")
        w_df = _read_optional(base / f"{slug}_ordinary_swd.csv")
        latent_w_df = _read_optional(base / f"{slug}_latent_swd.csv")
        swd_compare = _read_optional(base / f"{slug}_ordinary_vs_latent_swd.csv")
        period_err = _read_optional(base / f"{slug}_period_error_summary.csv")
        period_jsd = _read_optional(base / f"{slug}_period_jsd_summary.csv")
        period_swd = _read_optional(base / f"{slug}_period_swd.csv")
        losses_df = _read_optional(base / f"{slug}_ae_training_losses.csv")

        consensus_path = base / f"{slug}_consensus.csv"
        if consensus_path.exists():
            consensus = pd.read_csv(consensus_path)
            if "year" in consensus.columns:
                consensus = consensus.set_index("year")
        else:
            consensus = None

        ae_threshold = _threshold_from_period_summary(
            period_err, "Mean Error", "Std Error", std_multiplier
        )
        jsd_threshold = _threshold_from_period_summary(
            period_jsd, "Mean JSD", "Std JSD", std_multiplier
        )

        baseline = _reconstruct_baseline(base / f"{slug}_ordinary_swd_baseline_values.csv")
        latent_baseline = _reconstruct_baseline(base / f"{slug}_latent_swd_baseline_values.csv")

        results.append(
            {
                "name": name,
                "slug": slug,
                "train_end": train_end,
                "gap_end": gap_end,
                "ae_yearly": ae_yearly if ae_yearly is not None else pd.DataFrame(),
                "jsd_yearly": jsd_yearly if jsd_yearly is not None else pd.DataFrame(),
                "w_df": w_df if w_df is not None else pd.DataFrame(),
                "latent_w_df": latent_w_df if latent_w_df is not None else pd.DataFrame(),
                # extras for regeneration ------------------------------------
                "year_recon": ae_all,
                "year_jsd": jsd_all,
                "ks_df": ks_df,
                "ad_df": ad_df,
                "ae_threshold": ae_threshold,
                "jsd_threshold": jsd_threshold,
                "baseline": baseline,
                "latent_baseline": latent_baseline,
                "swd_compare": swd_compare,
                "summary": consensus,
                "period_error_summary": period_err,
                "period_jsd_summary": period_jsd,
                "period_swd": period_swd,
                "losses_df": losses_df,
                "table_dir": base,
            }
        )

    return results


def regenerate_per_encoder_figures(
    result: Dict[str, Any],
    output_dir: Path,
    config,
    *,
    rewrite_tex: bool = True,
) -> None:
    """Regenerate every per-encoder figure from a result dict produced by
    ``load_results_from_tables``. Also re-emits ``.tex`` siblings for the
    tables on disk if ``rewrite_tex=True`` (CSV content unchanged).

    ``config`` only needs ``figures_dir``, ``train_end``, ``gap_end``,
    ``drift_fraction_threshold`` and ``threshold_std_multiplier`` — pass
    a ``DriftConfig`` or any object exposing the same attributes.
    """
    name = result["name"]
    slug = result["slug"]
    fig_dir = ensure_dir(Path(output_dir) / "figures" / slug)

    # AE reconstruction error -------------------------------------------------
    year_recon = result.get("year_recon")
    ae_thr = result.get("ae_threshold")
    if year_recon is not None and not year_recon.empty and ae_thr is not None:
        plot_reconstruction_error(
            year_recon, ae_thr,
            f"{name} - Autoencoder reconstruction error over time",
            config, fig_dir / "ae_reconstruction_error_over_time.png",
        )
        plot_reconstruction_post_gap(
            year_recon, ae_thr,
            f"{name} - Autoencoder post-gap reconstruction error",
            config, fig_dir / "ae_reconstruction_error_post_gap.png",
        )

    # JSD --------------------------------------------------------------------
    year_jsd = result.get("year_jsd")
    jsd_thr = result.get("jsd_threshold")
    if year_jsd is not None and not year_jsd.empty and jsd_thr is not None:
        plot_jsd_over_time(
            year_jsd, jsd_thr,
            f"{name} - Jensen-Shannon divergence over time",
            config, fig_dir / "jsd_over_time.png",
        )
        plot_jsd_post_gap(
            year_jsd, jsd_thr,
            f"{name} - Jensen-Shannon divergence post-gap",
            config, fig_dir / "jsd_post_gap.png",
        )

    # KS / AD ----------------------------------------------------------------
    for method, stat_df in (("KS", result.get("ks_df")), ("AD", result.get("ad_df"))):
        if stat_df is None or stat_df.empty:
            continue
        lower = method.lower()
        plot_stat_fraction(stat_df, method, name, config,
                           fig_dir / f"{lower}_fraction_significant.png")
        plot_stat_value(stat_df, method, name, config,
                        fig_dir / f"{lower}_mean_statistic.png")
        plot_stat_post_gap(stat_df, method, name, config,
                           fig_dir / f"{lower}_post_gap_fraction.png")

    # Ordinary SWD -----------------------------------------------------------
    w_df = result.get("w_df")
    baseline = result.get("baseline")
    if w_df is not None and not w_df.empty and baseline is not None:
        plot_wasserstein_yearly(
            w_df, baseline,
            f"{name} - Ordinary SWD over time",
            config, fig_dir / "ordinary_swd_over_time.png",
        )
        plot_wasserstein_zscore(
            w_df, f"{name} - Ordinary SWD z-score",
            config, fig_dir / "ordinary_swd_zscore.png",
        )
        plot_baseline_histogram(
            baseline,
            f"{name} - Ordinary SWD bootstrap baseline",
            fig_dir / "ordinary_swd_baseline_histogram.png",
        )

    # Latent SWD -------------------------------------------------------------
    latent_w = result.get("latent_w_df")
    latent_bl = result.get("latent_baseline")
    if latent_w is not None and not latent_w.empty and latent_bl is not None:
        plot_wasserstein_yearly(
            latent_w, latent_bl,
            f"{name} - AE latent SWD over time",
            config, fig_dir / "latent_swd_over_time.png",
        )
        plot_wasserstein_zscore(
            latent_w, f"{name} - AE latent SWD z-score",
            config, fig_dir / "latent_swd_zscore.png",
        )
        plot_baseline_histogram(
            latent_bl,
            f"{name} - AE latent SWD bootstrap baseline",
            fig_dir / "latent_swd_baseline_histogram.png",
        )

    # Ordinary vs latent overlay ---------------------------------------------
    swd_compare = result.get("swd_compare")
    if swd_compare is not None and not swd_compare.empty \
       and "ordinary_swd" in swd_compare.columns \
       and "latent_swd" in swd_compare.columns:
        plot_two_lines(
            swd_compare, "year",
            "ordinary_swd", "latent_swd",
            "Ordinary SWD", "AE-latent SWD",
            f"{name} - Ordinary SWD vs AE-latent SWD",
            "Mean SWD",
            fig_dir / "ordinary_vs_latent_swd.png",
        )

    # Consensus --------------------------------------------------------------
    summary = result.get("summary")
    if summary is not None and not summary.empty:
        plot_consensus(
            summary, title=None,
            fig_path=fig_dir / "consensus_voting.png",
        )

    # Re-emit .tex siblings for every CSV (CSV content unchanged) ------------
    if rewrite_tex:
        table_dir = result.get("table_dir") or (Path(output_dir) / "tables" / slug)
        _rewrite_tex_for_encoder(result, Path(table_dir))


_HIGHLIGHTS: dict[str, tuple[str, str]] = {
    "ae_yearly":      ("mean", "max"),
    "jsd_yearly":     ("mean", "max"),
    "ordinary_swd":   ("mean_swd", "max"),
    "latent_swd":     ("mean_swd", "max"),
    "ks":             ("fraction_significant_fdr", "max"),
    "ad":             ("fraction_significant_fdr", "max"),
    "consensus":      ("Votes", "max"),
}


def _rewrite_tex_for_encoder(result: Dict[str, Any], table_dir: Path) -> None:
    """For every per-encoder table on disk, re-emit a booktabs .tex sibling.
    CSV content stays identical."""
    slug = result["slug"]
    name = result["name"]

    spec: list[tuple[str, pd.DataFrame | None, str, str, str | None]] = [
        ("ae_all_years",
            result.get("year_recon"),
            f"{name}: yearly autoencoder reconstruction error",
            f"tab:{slug}_ae_all_years", None),
        ("ae_yearly",
            result.get("ae_yearly"),
            f"{name}: post-gap autoencoder reconstruction error",
            f"tab:{slug}_ae_yearly", "ae_yearly"),
        ("jsd_all_years",
            result.get("year_jsd"),
            f"{name}: yearly Jensen-Shannon divergence",
            f"tab:{slug}_jsd_all_years", None),
        ("jsd_yearly",
            result.get("jsd_yearly"),
            f"{name}: post-gap Jensen-Shannon divergence",
            f"tab:{slug}_jsd_yearly", "jsd_yearly"),
        ("ks",
            result.get("ks_df"),
            f"{name}: Kolmogorov-Smirnov drift test",
            f"tab:{slug}_ks", "ks"),
        ("ad",
            result.get("ad_df"),
            f"{name}: Anderson-Darling drift test",
            f"tab:{slug}_ad", "ad"),
        ("ordinary_swd",
            result.get("w_df"),
            f"{name}: ordinary sliced Wasserstein distance",
            f"tab:{slug}_ordinary_swd", "ordinary_swd"),
        ("latent_swd",
            result.get("latent_w_df"),
            f"{name}: AE-latent sliced Wasserstein distance",
            f"tab:{slug}_latent_swd", "latent_swd"),
        ("ordinary_vs_latent_swd",
            result.get("swd_compare"),
            f"{name}: ordinary vs AE-latent SWD",
            f"tab:{slug}_ordinary_vs_latent_swd", None),
        ("period_error_summary",
            result.get("period_error_summary"),
            f"{name}: reconstruction error by period",
            f"tab:{slug}_period_error_summary", None),
        ("period_jsd_summary",
            result.get("period_jsd_summary"),
            f"{name}: Jensen-Shannon divergence by period",
            f"tab:{slug}_period_jsd_summary", None),
        ("period_swd",
            result.get("period_swd"),
            f"{name}: sliced Wasserstein distance by period",
            f"tab:{slug}_period_swd", None),
        ("ae_training_losses",
            result.get("losses_df"),
            f"{name}: autoencoder training loss per epoch",
            f"tab:{slug}_ae_training_losses", None),
    ]

    for suffix, df, caption, label, hl_key in spec:
        if df is None or df.empty:
            continue
        hl = _HIGHLIGHTS.get(hl_key) if hl_key else None
        save_table(
            df, table_dir / f"{slug}_{suffix}.csv",
            caption=caption, label=label, highlight=hl, index=False,
        )

    # Consensus is indexed by year; preserve the index in CSV + .tex.
    summary = result.get("summary")
    if summary is not None and not summary.empty:
        save_table(
            summary, table_dir / f"{slug}_consensus.csv",
            caption=f"{name}: drift voting consensus by year",
            label=f"tab:{slug}_consensus",
            highlight=_HIGHLIGHTS["consensus"], index=True,
        )

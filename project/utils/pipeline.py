from pathlib import Path

from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

from .config import DriftConfig
from .data import build_time_masks, parse_embedding_column, scale_embeddings
from .helpers import ensure_dir, slugify
from models.models import (
    compute_reconstruction_errors,
    compute_reconstruction_errors_and_jsd,
    compute_threshold,
    train_autoencoder,
)
from models.statistical_tests import (
    aggregate_metric_by_year,
    flag_drift,
    run_ad_test,
    run_ks_test,
)
from visualization.plot_theme import apply_paper_theme
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
from models.wasserstein import build_baseline, run_yearly_swd, sliced_wasserstein

apply_paper_theme()


def _save_table(df: pd.DataFrame, path: Path) -> None:
    """Back-compat shim: writes CSV + LaTeX (booktabs) sibling."""
    save_table(df, path, index=False)


def _save_series_table(df: pd.DataFrame, path: Path) -> None:
    """Back-compat shim for indexed frames (e.g. consensus by year)."""
    save_table(df, path, index=True, highlight=("Votes", "max"))


def run_full_pipeline(
    df_input: pd.DataFrame,
    emb_col: str,
    pretty_name: str | None,
    config: DriftConfig,
) -> Dict[str, Any]:
    if pretty_name is None:
        pretty_name = emb_col

    slug = slugify(pretty_name)

    fig_dir = ensure_dir(config.figures_dir / slug)
    table_dir = ensure_dir(config.tables_dir / slug)
    model_dir = ensure_dir(config.models_dir / slug)

    print("=" * 80)
    print(f"RUNNING PIPELINE FOR: {pretty_name}")
    print("=" * 80)

    df = df_input.copy()

    X = parse_embedding_column(df[emb_col])

    train_mask, gap_mask, test_mask = build_time_masks(df, config)

    print(f"Total reviews       : {len(df):,}")
    print(f"Train <= {config.train_end}: {train_mask.sum():,}")
    print(f"Gap {config.train_end + 1}-{config.gap_end}: {gap_mask.sum():,}")
    print(f"Test > {config.gap_end}: {test_mask.sum():,}")
    print(f"Embedding dimension : {X.shape[1]}")

    X_train_scaled, X_scaled, scaler = scale_embeddings(X, train_mask)

    joblib.dump(scaler, model_dir / "scaler.joblib")

    model, losses = train_autoencoder(
        X_train_scaled,
        input_dim=X.shape[1],
        config=config,
    )

    torch.save(
        model.state_dict(),
        model_dir / "autoencoder_state_dict.pt",
    )

    errors, latent, jsd = compute_reconstruction_errors_and_jsd(model, X_scaled)

    df["reconstruction_error"] = errors
    df["js_divergence"] = jsd

    year_recon = aggregate_metric_by_year(df, "reconstruction_error")
    year_jsd = aggregate_metric_by_year(df, "js_divergence")

    train_err = df.loc[train_mask, "reconstruction_error"].values

    threshold = compute_threshold(
        train_err,
        config.threshold_std_multiplier,
    )

    print(f"\nAE training error mean: {train_err.mean():.6f}")
    print(f"AE training error std : {train_err.std():.6f}")
    print(f"AE threshold std mult : {config.threshold_std_multiplier:.3f}")
    print(f"AE threshold          : {threshold:.6f}")

    train_jsd = df.loc[train_mask, "js_divergence"].values
    jsd_threshold = compute_threshold(
        train_jsd,
        config.threshold_std_multiplier,
    )
    print(f"\nJSD training mean     : {train_jsd.mean():.6f}")
    print(f"JSD training std      : {train_jsd.std():.6f}")
    print(f"JSD threshold         : {jsd_threshold:.6f}")

    plot_reconstruction_error(
        year_recon,
        threshold,
        f"{pretty_name} - Autoencoder reconstruction error over time",
        config,
        fig_dir / "ae_reconstruction_error_over_time.png",
    )

    plot_reconstruction_post_gap(
        year_recon,
        threshold,
        f"{pretty_name} - Autoencoder post-gap reconstruction error",
        config,
        fig_dir / "ae_reconstruction_error_post_gap.png",
    )

    plot_jsd_over_time(
        year_jsd,
        jsd_threshold,
        f"{pretty_name} - Jensen-Shannon divergence over time",
        config,
        fig_dir / "jsd_over_time.png",
    )

    plot_jsd_post_gap(
        year_jsd,
        jsd_threshold,
        f"{pretty_name} - Jensen-Shannon divergence post-gap",
        config,
        fig_dir / "jsd_post_gap.png",
    )

    comparison = pd.DataFrame(
        {
            "Period": [
                f"Train <= {config.train_end}",
                f"Gap {config.train_end + 1}-{config.gap_end}",
                f"Test > {config.gap_end}",
            ],
            "Samples": [
                train_mask.sum(),
                gap_mask.sum(),
                test_mask.sum(),
            ],
            "Mean Error": [
                df.loc[train_mask, "reconstruction_error"].mean(),
                df.loc[gap_mask, "reconstruction_error"].mean()
                if gap_mask.any()
                else np.nan,
                df.loc[test_mask, "reconstruction_error"].mean()
                if test_mask.any()
                else np.nan,
            ],
            "Std Error": [
                df.loc[train_mask, "reconstruction_error"].std(),
                df.loc[gap_mask, "reconstruction_error"].std()
                if gap_mask.any()
                else np.nan,
                df.loc[test_mask, "reconstruction_error"].std()
                if test_mask.any()
                else np.nan,
            ],
        }
    )

    jsd_comparison = pd.DataFrame(
        {
            "Period": [
                f"Train <= {config.train_end}",
                f"Gap {config.train_end + 1}-{config.gap_end}",
                f"Test > {config.gap_end}",
            ],
            "Samples": [
                train_mask.sum(),
                gap_mask.sum(),
                test_mask.sum(),
            ],
            "Mean JSD": [
                df.loc[train_mask, "js_divergence"].mean(),
                df.loc[gap_mask, "js_divergence"].mean()
                if gap_mask.any()
                else np.nan,
                df.loc[test_mask, "js_divergence"].mean()
                if test_mask.any()
                else np.nan,
            ],
            "Std JSD": [
                df.loc[train_mask, "js_divergence"].std(),
                df.loc[gap_mask, "js_divergence"].std()
                if gap_mask.any()
                else np.nan,
                df.loc[test_mask, "js_divergence"].std()
                if test_mask.any()
                else np.nan,
            ],
        }
    )

    ae_yearly = year_recon[year_recon["year"] > config.gap_end].copy()
    ae_yearly["drift_flag"] = ae_yearly["mean"] > threshold

    jsd_yearly = year_jsd[year_jsd["year"] > config.gap_end].copy()
    jsd_yearly["drift_flag"] = jsd_yearly["mean"] > jsd_threshold

    ks_df = flag_drift(
        run_ks_test(df, X_scaled, train_mask, config),
        config,
    )

    ad_df = flag_drift(
        run_ad_test(df, X_scaled, train_mask, config),
        config,
    )

    for method_name, stat_df in [("KS", ks_df), ("AD", ad_df)]:
        lower = method_name.lower()

        plot_stat_fraction(
            stat_df,
            method_name,
            pretty_name,
            config,
            fig_dir / f"{lower}_fraction_significant.png",
        )

        plot_stat_value(
            stat_df,
            method_name,
            pretty_name,
            config,
            fig_dir / f"{lower}_mean_statistic.png",
        )

        plot_stat_post_gap(
            stat_df,
            method_name,
            pretty_name,
            config,
            fig_dir / f"{lower}_post_gap_fraction.png",
        )

    baseline = build_baseline(
        X_train_scaled,
        config,
        random_state=config.seed,
    )

    w_df = run_yearly_swd(
        df,
        X_scaled,
        X_train_scaled,
        baseline,
        config,
    )

    plot_wasserstein_yearly(
        w_df,
        baseline,
        f"{pretty_name} - Ordinary SWD over time",
        config,
        fig_dir / "ordinary_swd_over_time.png",
    )

    plot_wasserstein_zscore(
        w_df,
        f"{pretty_name} - Ordinary SWD z-score",
        config,
        fig_dir / "ordinary_swd_zscore.png",
    )

    plot_baseline_histogram(
        baseline,
        f"{pretty_name} - Ordinary SWD bootstrap baseline",
        fig_dir / "ordinary_swd_baseline_histogram.png",
    )

    X_gap_scaled = X_scaled[gap_mask]
    X_test_scaled = X_scaled[test_mask]

    period_swd_rows = []

    if len(X_gap_scaled) > 0:
        gap_swd = sliced_wasserstein(
            X_train_scaled,
            X_gap_scaled,
            config.swd_n_projections,
            random_state=123,
            max_samples=config.swd_max_samples,
        )

        period_swd_rows.append(
            {
                "period": "train_vs_gap",
                "mean_swd": gap_swd["mean_swd"],
                "drift_95": gap_swd["mean_swd"] > baseline["threshold_95"],
            }
        )

    if len(X_test_scaled) > 0:
        test_swd = sliced_wasserstein(
            X_train_scaled,
            X_test_scaled,
            config.swd_n_projections,
            random_state=456,
            max_samples=config.swd_max_samples,
        )

        period_swd_rows.append(
            {
                "period": "train_vs_test",
                "mean_swd": test_swd["mean_swd"],
                "drift_95": test_swd["mean_swd"] > baseline["threshold_95"],
            }
        )

    period_swd = pd.DataFrame(period_swd_rows)

    latent_train = latent[train_mask]

    latent_baseline = build_baseline(
        latent_train,
        config,
        random_state=config.seed,
    )

    latent_w_df = run_yearly_swd(
        df,
        latent,
        latent_train,
        latent_baseline,
        config,
    )

    plot_wasserstein_yearly(
        latent_w_df,
        latent_baseline,
        f"{pretty_name} - AE latent SWD over time",
        config,
        fig_dir / "latent_swd_over_time.png",
    )

    plot_wasserstein_zscore(
        latent_w_df,
        f"{pretty_name} - AE latent SWD z-score",
        config,
        fig_dir / "latent_swd_zscore.png",
    )

    plot_baseline_histogram(
        latent_baseline,
        f"{pretty_name} - AE latent SWD bootstrap baseline",
        fig_dir / "latent_swd_baseline_histogram.png",
    )

    swd_compare = (
        w_df[["year", "mean_swd", "z_score", "drift_95"]]
        .rename(
            columns={
                "mean_swd": "ordinary_swd",
                "z_score": "ordinary_z",
                "drift_95": "ordinary_drift_95",
            }
        )
        .merge(
            latent_w_df[["year", "mean_swd", "z_score", "drift_95"]].rename(
                columns={
                    "mean_swd": "latent_swd",
                    "z_score": "latent_z",
                    "drift_95": "latent_drift_95",
                }
            ),
            on="year",
            how="outer",
        )
    )

    plot_two_lines(
        swd_compare,
        "year",
        "ordinary_swd",
        "latent_swd",
        "Ordinary SWD",
        "AE-latent SWD",
        f"{pretty_name} - Ordinary SWD vs AE-latent SWD",
        "Mean SWD",
        fig_dir / "ordinary_vs_latent_swd.png",
    )

    ae_flags = ae_yearly.set_index("year")["drift_flag"].rename("AE")
    jsd_flags = jsd_yearly.set_index("year")["drift_flag"].rename("JSD")
    ks_flags = ks_df.set_index("year")["drift_detected"].rename("KS")
    ad_flags = ad_df.set_index("year")["drift_detected"].rename("AD")
    swd_flags = w_df.set_index("year")["drift_95"].rename("SWD")

    summary = pd.concat(
        [ae_flags, jsd_flags, ks_flags, ad_flags, swd_flags],
        axis=1,
    )

    summary = summary.loc[summary.index > config.gap_end]
    summary["Votes"] = summary[["AE", "JSD", "KS", "AD", "SWD"]].sum(axis=1).astype(int)
    summary["Consensus"] = summary["Votes"] >= 3

    plot_consensus(
        summary,
        title=None,
        fig_path=fig_dir / "consensus_voting.png",
    )

    losses_df = pd.DataFrame(
        {
            "epoch": range(1, len(losses) + 1),
            "loss": losses,
        }
    )

    baseline_df = pd.DataFrame(
        {
            "bootstrap_swd": baseline["values"],
        }
    )

    latent_baseline_df = pd.DataFrame(
        {
            "bootstrap_swd": latent_baseline["values"],
        }
    )

    _save_table(year_recon, table_dir / f"{slug}_ae_all_years.csv")
    _save_table(ae_yearly, table_dir / f"{slug}_ae_yearly.csv")
    _save_table(comparison, table_dir / f"{slug}_period_error_summary.csv")
    _save_table(year_jsd, table_dir / f"{slug}_jsd_all_years.csv")
    _save_table(jsd_yearly, table_dir / f"{slug}_jsd_yearly.csv")
    _save_table(jsd_comparison, table_dir / f"{slug}_period_jsd_summary.csv")
    _save_table(ks_df, table_dir / f"{slug}_ks.csv")
    _save_table(ad_df, table_dir / f"{slug}_ad.csv")
    _save_table(w_df, table_dir / f"{slug}_ordinary_swd.csv")
    _save_table(latent_w_df, table_dir / f"{slug}_latent_swd.csv")
    _save_table(swd_compare, table_dir / f"{slug}_ordinary_vs_latent_swd.csv")
    _save_series_table(summary, table_dir / f"{slug}_consensus.csv")
    _save_table(losses_df, table_dir / f"{slug}_ae_training_losses.csv")
    _save_table(period_swd, table_dir / f"{slug}_period_swd.csv")
    _save_table(baseline_df, table_dir / f"{slug}_ordinary_swd_baseline_values.csv")
    _save_table(
        latent_baseline_df,
        table_dir / f"{slug}_latent_swd_baseline_values.csv",
    )

    return {
        "name": pretty_name,
        "slug": slug,
        "train_end": config.train_end,
        "gap_end": config.gap_end,
        "df": df,
        "X": X,
        "X_scaled": X_scaled,
        "X_train_scaled": X_train_scaled,
        "train_mask": train_mask,
        "gap_mask": gap_mask,
        "test_mask": test_mask,
        "model": model,
        "losses": losses,
        "latent": latent,
        "year_recon": year_recon,
        "ae_yearly": ae_yearly,
        "year_jsd": year_jsd,
        "jsd_yearly": jsd_yearly,
        "ks_df": ks_df,
        "ad_df": ad_df,
        "baseline": baseline,
        "w_df": w_df,
        "latent_baseline": latent_baseline,
        "latent_w_df": latent_w_df,
        "swd_compare": swd_compare,
        "summary": summary,
    }


def run_autoencoder_swd_pipeline(
    df_input: pd.DataFrame,
    emb_col: str,
    pretty_name: str | None,
    config: DriftConfig,
) -> Dict[str, Any]:
    if pretty_name is None:
        pretty_name = emb_col

    slug = slugify(pretty_name)

    fig_dir = ensure_dir(config.figures_dir / slug)
    table_dir = ensure_dir(config.tables_dir / slug)
    model_dir = ensure_dir(config.models_dir / slug)

    print("=" * 80)
    print(f"RUNNING AUTOENCODER + SLICED WASSERSTEIN FOR: {pretty_name}")
    print("=" * 80)

    df = df_input.copy()
    X = parse_embedding_column(df[emb_col])
    train_mask, gap_mask, test_mask = build_time_masks(df, config)

    print(f"Total reviews       : {len(df):,}")
    print(f"Train <= {config.train_end}: {train_mask.sum():,}")
    print(f"Gap {config.train_end + 1}-{config.gap_end}: {gap_mask.sum():,}")
    print(f"Test > {config.gap_end}: {test_mask.sum():,}")
    print(f"Embedding dimension : {X.shape[1]}")
    print(
        "Autoencoder layers  : "
        f"{X.shape[1]} -> {config.hidden_1} -> {config.hidden_2} -> "
        f"{config.hidden_3} -> {config.hidden_4} -> {config.hidden_5} -> "
        f"{config.latent_dim}"
    )
    print(f"SWD projections     : {config.swd_n_projections}")
    print(f"SWD bootstrap runs  : {config.swd_bootstrap_runs}")
    print(f"SWD max samples     : {config.swd_max_samples:,} per distribution")

    X_train_scaled, X_scaled, scaler = scale_embeddings(X, train_mask)
    joblib.dump(scaler, model_dir / "scaler.joblib")

    model, losses = train_autoencoder(
        X_train_scaled,
        input_dim=X.shape[1],
        config=config,
    )

    torch.save(
        model.state_dict(),
        model_dir / "autoencoder_state_dict.pt",
    )

    errors, latent = compute_reconstruction_errors(model, X_scaled)

    df["reconstruction_error"] = errors
    year_recon = aggregate_metric_by_year(df, "reconstruction_error")
    ae_yearly = year_recon[year_recon["year"] > config.gap_end].copy()

    train_err = df.loc[train_mask, "reconstruction_error"].values
    threshold = compute_threshold(
        train_err,
        config.threshold_std_multiplier,
    )
    ae_yearly["drift_flag"] = ae_yearly["mean"] > threshold

    print(f"\nAE training error mean: {train_err.mean():.6f}")
    print(f"AE training error std : {train_err.std():.6f}")
    print(f"AE threshold std mult : {config.threshold_std_multiplier:.3f}")
    print(f"AE threshold          : {threshold:.6f}")

    plot_reconstruction_error(
        year_recon,
        threshold,
        f"{pretty_name} - Autoencoder reconstruction error over time",
        config,
        fig_dir / "ae_reconstruction_error_over_time.png",
    )

    plot_reconstruction_post_gap(
        year_recon,
        threshold,
        f"{pretty_name} - Autoencoder post-gap reconstruction error",
        config,
        fig_dir / "ae_reconstruction_error_post_gap.png",
    )

    baseline = build_baseline(
        X_train_scaled,
        config,
        random_state=config.seed,
    )

    w_df = run_yearly_swd(
        df,
        X_scaled,
        X_train_scaled,
        baseline,
        config,
    )

    plot_wasserstein_yearly(
        w_df,
        baseline,
        f"{pretty_name} - Ordinary SWD over time",
        config,
        fig_dir / "ordinary_swd_over_time.png",
    )

    plot_wasserstein_zscore(
        w_df,
        f"{pretty_name} - Ordinary SWD z-score",
        config,
        fig_dir / "ordinary_swd_zscore.png",
    )

    plot_baseline_histogram(
        baseline,
        f"{pretty_name} - Ordinary SWD bootstrap baseline",
        fig_dir / "ordinary_swd_baseline_histogram.png",
    )

    X_gap_scaled = X_scaled[gap_mask]
    X_test_scaled = X_scaled[test_mask]
    period_swd_rows = []

    if len(X_gap_scaled) > 0:
        gap_swd = sliced_wasserstein(
            X_train_scaled,
            X_gap_scaled,
            config.swd_n_projections,
            random_state=123,
            max_samples=config.swd_max_samples,
        )
        period_swd_rows.append(
            {
                "period": "train_vs_gap",
                "mean_swd": gap_swd["mean_swd"],
                "drift_95": gap_swd["mean_swd"] > baseline["threshold_95"],
            }
        )

    if len(X_test_scaled) > 0:
        test_swd = sliced_wasserstein(
            X_train_scaled,
            X_test_scaled,
            config.swd_n_projections,
            random_state=456,
            max_samples=config.swd_max_samples,
        )
        period_swd_rows.append(
            {
                "period": "train_vs_test",
                "mean_swd": test_swd["mean_swd"],
                "drift_95": test_swd["mean_swd"] > baseline["threshold_95"],
            }
        )

    period_swd = pd.DataFrame(period_swd_rows)

    latent_train = latent[train_mask]
    latent_baseline = build_baseline(
        latent_train,
        config,
        random_state=config.seed,
    )

    latent_w_df = run_yearly_swd(
        df,
        latent,
        latent_train,
        latent_baseline,
        config,
    )

    plot_wasserstein_yearly(
        latent_w_df,
        latent_baseline,
        f"{pretty_name} - AE latent SWD over time",
        config,
        fig_dir / "latent_swd_over_time.png",
    )

    plot_wasserstein_zscore(
        latent_w_df,
        f"{pretty_name} - AE latent SWD z-score",
        config,
        fig_dir / "latent_swd_zscore.png",
    )

    plot_baseline_histogram(
        latent_baseline,
        f"{pretty_name} - AE latent SWD bootstrap baseline",
        fig_dir / "latent_swd_baseline_histogram.png",
    )

    swd_compare = (
        w_df[["year", "mean_swd", "z_score", "drift_95"]]
        .rename(
            columns={
                "mean_swd": "ordinary_swd",
                "z_score": "ordinary_z",
                "drift_95": "ordinary_drift_95",
            }
        )
        .merge(
            latent_w_df[["year", "mean_swd", "z_score", "drift_95"]].rename(
                columns={
                    "mean_swd": "latent_swd",
                    "z_score": "latent_z",
                    "drift_95": "latent_drift_95",
                }
            ),
            on="year",
            how="outer",
        )
    )

    plot_two_lines(
        swd_compare,
        "year",
        "ordinary_swd",
        "latent_swd",
        "Ordinary SWD",
        "AE-latent SWD",
        f"{pretty_name} - Ordinary SWD vs AE-latent SWD",
        "Mean SWD",
        fig_dir / "ordinary_vs_latent_swd.png",
    )

    losses_df = pd.DataFrame(
        {
            "epoch": range(1, len(losses) + 1),
            "loss": losses,
        }
    )
    baseline_df = pd.DataFrame({"bootstrap_swd": baseline["values"]})
    latent_baseline_df = pd.DataFrame({"bootstrap_swd": latent_baseline["values"]})

    _save_table(year_recon, table_dir / f"{slug}_ae_all_years.csv")
    _save_table(ae_yearly, table_dir / f"{slug}_ae_yearly.csv")
    _save_table(losses_df, table_dir / f"{slug}_ae_training_losses.csv")
    _save_table(w_df, table_dir / f"{slug}_ordinary_swd.csv")
    _save_table(latent_w_df, table_dir / f"{slug}_latent_swd.csv")
    _save_table(swd_compare, table_dir / f"{slug}_ordinary_vs_latent_swd.csv")
    _save_table(period_swd, table_dir / f"{slug}_period_swd.csv")
    _save_table(baseline_df, table_dir / f"{slug}_ordinary_swd_baseline_values.csv")
    _save_table(
        latent_baseline_df,
        table_dir / f"{slug}_latent_swd_baseline_values.csv",
    )

    return {
        "name": pretty_name,
        "slug": slug,
        "train_end": config.train_end,
        "gap_end": config.gap_end,
        "df": df,
        "X": X,
        "X_scaled": X_scaled,
        "X_train_scaled": X_train_scaled,
        "train_mask": train_mask,
        "gap_mask": gap_mask,
        "test_mask": test_mask,
        "model": model,
        "losses": losses,
        "latent": latent,
        "year_recon": year_recon,
        "ae_yearly": ae_yearly,
        "baseline": baseline,
        "w_df": w_df,
        "latent_baseline": latent_baseline,
        "latent_w_df": latent_w_df,
        "swd_compare": swd_compare,
    }

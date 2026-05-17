import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp, ks_2samp
from statsmodels.stats.multitest import multipletests

from utils.config import DriftConfig


def aggregate_metric_by_year(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    return (
        df.groupby("year")[metric_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def aggregate_errors_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_metric_by_year(df, "reconstruction_error")


def run_ks_test(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    train_mask: np.ndarray,
    config: DriftConfig,
) -> pd.DataFrame:
    X_train = X_scaled[train_mask]
    rows = []

    for year in sorted(df["year"].unique()):
        if year <= config.train_end:
            continue

        idx = df["year"].values == year
        X_year = X_scaled[idx]

        if len(X_year) < config.min_samples_per_year:
            continue

        stats_, pvals_ = [], []

        for j in range(X_scaled.shape[1]):
            stat, pval = ks_2samp(
                X_train[:, j],
                X_year[:, j],
                alternative="two-sided",
                mode="auto",
            )
            stats_.append(stat)
            pvals_.append(pval)

        stats_ = np.array(stats_)
        pvals_ = np.array(pvals_)

        reject, _, _, _ = multipletests(
            pvals_,
            alpha=config.alpha,
            method="fdr_bh",
        )

        rows.append(
            {
                "year": year,
                "n_samples": len(X_year),
                "fraction_significant_raw": float(np.mean(pvals_ < config.alpha)),
                "fraction_significant_fdr": float(np.mean(reject)),
                "mean_stat": float(np.mean(stats_)),
                "median_stat": float(np.median(stats_)),
            }
        )

    return pd.DataFrame(rows)


def run_ad_test(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    train_mask: np.ndarray,
    config: DriftConfig,
) -> pd.DataFrame:
    X_train = X_scaled[train_mask]
    rows = []

    for year in sorted(df["year"].unique()):
        if year <= config.train_end:
            continue

        idx = df["year"].values == year
        X_year = X_scaled[idx]

        if len(X_year) < config.min_samples_per_year:
            continue

        stats_, pvals_proxy = [], []

        for j in range(X_scaled.shape[1]):
            try:
                res = anderson_ksamp([X_train[:, j], X_year[:, j]])

                stats_.append(float(res.statistic))

                p_proxy = min(
                    max(res.significance_level / 100.0, 0.001),
                    0.25,
                )
                pvals_proxy.append(p_proxy)

            except Exception:
                stats_.append(np.nan)
                pvals_proxy.append(1.0)

        stats_ = np.array(stats_)
        pvals_proxy = np.array(pvals_proxy)

        valid = ~np.isnan(stats_)
        stats_valid = stats_[valid]
        pvals_valid = pvals_proxy[valid]

        reject, _, _, _ = multipletests(
            pvals_valid,
            alpha=config.alpha,
            method="fdr_bh",
        )

        rows.append(
            {
                "year": year,
                "n_samples": len(X_year),
                "fraction_significant_raw": float(np.mean(pvals_valid < config.alpha)),
                "fraction_significant_fdr": float(np.mean(reject)),
                "mean_stat": float(np.mean(stats_valid)),
                "median_stat": float(np.median(stats_valid)),
            }
        )

    return pd.DataFrame(rows)


def flag_drift(
    stat_df: pd.DataFrame,
    config: DriftConfig,
) -> pd.DataFrame:
    stat_df = stat_df.copy()

    stat_df["drift_detected"] = (
        stat_df["fraction_significant_fdr"] > config.drift_fraction_threshold
    )

    return stat_df

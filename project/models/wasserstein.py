import numpy as np
import pandas as pd

from utils.config import DriftConfig


def wasserstein_1d_sorted(x, y) -> float:
    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))

    n = min(len(x), len(y))

    if n == 0:
        return np.nan

    if len(x) != n:
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        x = x[idx]

    if len(y) != n:
        idx = np.linspace(0, len(y) - 1, n).astype(int)
        y = y[idx]

    return float(np.mean(np.abs(x - y)))


def sliced_wasserstein(
    X_ref,
    X_cmp,
    n_projections: int = 256,
    random_state: int = 42,
    max_samples: int | None = None,
) -> dict:
    rng = np.random.default_rng(random_state)

    if max_samples is not None:
        if len(X_ref) > max_samples:
            ref_idx = rng.choice(len(X_ref), size=max_samples, replace=False)
            X_ref = X_ref[ref_idx]

        if len(X_cmp) > max_samples:
            cmp_idx = rng.choice(len(X_cmp), size=max_samples, replace=False)
            X_cmp = X_cmp[cmp_idx]

    d = X_ref.shape[1]

    projections = rng.normal(size=(n_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12

    dists = []

    for p in projections:
        ref_proj = X_ref @ p
        cmp_proj = X_cmp @ p

        dists.append(wasserstein_1d_sorted(ref_proj, cmp_proj))

    dists = np.array(dists)

    return {
        "mean_swd": float(np.mean(dists)),
        "std_swd": float(np.std(dists)),
        "all_swd": dists,
    }


def build_baseline(
    X_train_scaled,
    config: DriftConfig,
    random_state: int | None = None,
) -> dict:
    if random_state is None:
        random_state = config.seed

    rng = np.random.default_rng(random_state)
    values = []

    n = len(X_train_scaled)
    sample_size = min(config.swd_max_samples, n // 2)
    if sample_size == 0:
        raise ValueError("Need at least 2 training samples to build SWD baseline")

    print(
        "Building SWD baseline "
        f"({config.swd_bootstrap_runs} runs, "
        f"{config.swd_n_projections} projections, "
        f"{sample_size:,} samples per split)..."
    )

    for i in range(config.swd_bootstrap_runs):
        idx = rng.choice(n, size=2 * sample_size, replace=False)

        a = X_train_scaled[idx[:sample_size]]
        b = X_train_scaled[idx[sample_size:]]

        swd = sliced_wasserstein(
            a,
            b,
            n_projections=config.swd_n_projections,
            random_state=random_state + i,
        )

        values.append(swd["mean_swd"])

        if (i + 1) % 10 == 0 or i + 1 == config.swd_bootstrap_runs:
            print(f"  SWD baseline run {i + 1}/{config.swd_bootstrap_runs}")

    values = np.array(values)

    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "threshold_95": float(np.percentile(values, 95)),
        "threshold_99": float(np.percentile(values, 99)),
    }


def run_yearly_swd(
    df: pd.DataFrame,
    X_scaled,
    X_train_scaled,
    baseline: dict,
    config: DriftConfig,
) -> pd.DataFrame:
    rows = []

    for year in sorted(df["year"].unique()):
        if year <= config.train_end:
            continue

        idx = df["year"].values == year
        X_year = X_scaled[idx]

        if len(X_year) < config.min_samples_per_year:
            continue

        swd = sliced_wasserstein(
            X_train_scaled,
            X_year,
            n_projections=config.swd_n_projections,
            random_state=config.seed + int(year),
            max_samples=config.swd_max_samples,
        )

        z = (swd["mean_swd"] - baseline["mean"]) / (baseline["std"] + 1e-12)

        rows.append(
            {
                "year": int(year),
                "n_samples": len(X_year),
                "mean_swd": swd["mean_swd"],
                "std_swd": swd["std_swd"],
                "z_score": float(z),
                "drift_95": bool(swd["mean_swd"] > baseline["threshold_95"]),
                "drift_99": bool(swd["mean_swd"] > baseline["threshold_99"]),
            }
        )

    return pd.DataFrame(rows)

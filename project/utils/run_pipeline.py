import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.comparison import compare_all_embeddings
from utils.config import DriftConfig
from utils.data import load_reviews
from utils.helpers import set_seed
from utils.pipeline import run_full_pipeline
from visualization.plot_theme import apply_paper_theme


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run semantic drift detection on embedding columns."
    )

    parser.add_argument(
        "--input",
        default="data/reviews_with_3_embedding_types.csv",
        help="Path to single CSV file that contains all embedding columns",
    )

    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help="List of CSV files, one per encoding (same order as --names)",
    )

    parser.add_argument(
        "--embedding-cols",
        nargs="+",
        default=["emb_minilm", "emb_labse", "emb_distilbert"],
        help="Embedding columns to analyze when using --input",
    )

    parser.add_argument(
        "--embedding-prefix",
        default="embedding_",
        help="Prefix for wide embedding columns when using --input-files",
    )

    parser.add_argument(
        "--names",
        nargs="+",
        default=["MiniLM", "LaBSE", "DistilBERT"],
        help="Pretty names for embedding columns",
    )

    parser.add_argument(
        "--date-col",
        default="review_date",
        help="Date column used to create year",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for tables, figures, models, logs",
    )

    parser.add_argument("--train-end", type=int, default=2005)
    parser.add_argument("--gap-end", type=int, default=2010)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-1", type=int, default=512)
    parser.add_argument("--hidden-2", type=int, default=384)
    parser.add_argument("--hidden-3", type=int, default=256)
    parser.add_argument("--hidden-4", type=int, default=128)
    parser.add_argument("--hidden-5", type=int, default=96)

    parser.add_argument("--swd-projections", type=int, default=512)
    parser.add_argument("--swd-bootstrap-runs", type=int, default=200)
    parser.add_argument("--swd-max-samples", type=int, default=10_000)
    parser.add_argument("--threshold-std-multiplier", type=float, default=0.5)

    parser.add_argument("--min-samples-per-year", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DriftConfig(
        train_end=args.train_end,
        gap_end=args.gap_end,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        hidden_3=args.hidden_3,
        hidden_4=args.hidden_4,
        hidden_5=args.hidden_5,
        swd_n_projections=args.swd_projections,
        swd_bootstrap_runs=args.swd_bootstrap_runs,
        swd_max_samples=args.swd_max_samples,
        min_samples_per_year=args.min_samples_per_year,
        seed=args.seed,
        threshold_std_multiplier=args.threshold_std_multiplier,
        output_dir=Path(args.output_dir),
    )

    config.create_output_dirs()
    set_seed(config.seed)
    apply_paper_theme()

    results = []

    # Mode 1: one CSV per encoding (wide embedding columns like embedding_0..embedding_n)
    if args.input_files:
        if args.names is not None and len(args.names) != len(args.input_files):
            raise ValueError("--names must have the same number of values as --input-files")

        for idx, file_path in enumerate(args.input_files):
            pretty_name = args.names[idx] if args.names else Path(file_path).stem
            df = load_reviews(
                file_path,
                date_col=args.date_col,
            )
            print(
                f"Loaded data ({pretty_name}): {df.shape[0]:,} rows, {df.shape[1]:,} columns"
            )
            print(f"Year range: {df['year'].min()} - {df['year'].max()}")

            emb_cols = [
                col for col in df.columns if col.startswith(args.embedding_prefix)
            ]
            if not emb_cols:
                raise ValueError(
                    f"No embedding columns found in {file_path} with prefix '{args.embedding_prefix}'"
                )

            emb_cols = sorted(
                emb_cols,
                key=lambda x: int(x[len(args.embedding_prefix) :])
                if x[len(args.embedding_prefix) :].isdigit()
                else np.inf,
            )
            df["__embedding__"] = df[emb_cols].values.tolist()

            result = run_full_pipeline(
                df,
                emb_col="__embedding__",
                pretty_name=pretty_name,
                config=config,
            )
            results.append(result)
    # Mode 2: single combined CSV with one column per encoding
    else:
        if args.names is not None and len(args.names) != len(args.embedding_cols):
            raise ValueError(
                "--names must have the same number of values as --embedding-cols"
            )

        df = load_reviews(
            args.input,
            date_col=args.date_col,
        )

        print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")

        for idx, emb_col in enumerate(args.embedding_cols):
            if emb_col not in df.columns:
                raise ValueError(f"Embedding column not found in input data: {emb_col}")

            pretty_name = args.names[idx] if args.names else emb_col

            result = run_full_pipeline(
                df,
                emb_col=emb_col,
                pretty_name=pretty_name,
                config=config,
            )

            results.append(result)

    compare_all_embeddings(
        results,
        Path(args.output_dir),
    )

    print("\nDone.")
    print("Check:")
    print("- outputs/tables")
    print("- outputs/figures")
    print("- outputs/models")


if __name__ == "__main__":
    main()

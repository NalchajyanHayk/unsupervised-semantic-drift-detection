import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DriftConfig
from utils.data import load_reviews
from utils.helpers import set_seed
from utils.pipeline import run_autoencoder_swd_pipeline
from visualization.plot_theme import apply_paper_theme


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only autoencoder and sliced Wasserstein models on encoded data."
    )

    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
            "data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv",
            "data/encoded/reviews_encoded_LaBSE.csv",
            "data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv",
        ],
        help="Encoded CSV files to process, one per encoder.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["MiniLM", "LaBSE", "DistilBERT"],
        help="Pretty names for the encoded files.",
    )
    parser.add_argument(
        "--embedding-prefix",
        default="embedding_",
        help="Prefix for wide embedding columns in each encoded CSV.",
    )
    parser.add_argument(
        "--date-col",
        default="review_date",
        help="Date column used to create year.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for tables, figures, models, and logs.",
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

    if len(args.names) != len(args.input_files):
        raise ValueError("--names must have the same number of values as --input-files")

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

    for idx, file_path in enumerate(args.input_files):
        pretty_name = args.names[idx]
        df = load_reviews(
            file_path,
            date_col=args.date_col,
        )

        print(
            f"Loaded data ({pretty_name}): {df.shape[0]:,} rows, {df.shape[1]:,} columns"
        )
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")

        emb_cols = [col for col in df.columns if col.startswith(args.embedding_prefix)]
        if not emb_cols:
            raise ValueError(
                f"No embedding columns found in {file_path} with prefix "
                f"'{args.embedding_prefix}'"
            )

        emb_cols = sorted(
            emb_cols,
            key=lambda x: int(x[len(args.embedding_prefix) :])
            if x[len(args.embedding_prefix) :].isdigit()
            else np.inf,
        )
        df["__embedding__"] = df[emb_cols].values.tolist()

        run_autoencoder_swd_pipeline(
            df,
            emb_col="__embedding__",
            pretty_name=pretty_name,
            config=config,
        )

    print("\nDone.")
    print("Generated only autoencoder and sliced Wasserstein outputs.")
    print("- outputs/tables")
    print("- outputs/figures")
    print("- outputs/models")


if __name__ == "__main__":
    main()

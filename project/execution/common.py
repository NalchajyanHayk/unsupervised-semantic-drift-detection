"""Shared CLI helpers for execution-layer scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.encoder_pipeline import preprocess_and_encode
from utils.config import DriftConfig
from utils.helpers import set_seed
from visualization.plot_theme import apply_paper_theme


DEFAULT_INPUT_FILES = [
    "data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv",
    "data/encoded/reviews_encoded_LaBSE.csv",
    "data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv",
]

DEFAULT_NAMES = ["MiniLM", "LaBSE", "DistilBERT"]


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(DEFAULT_INPUT_FILES),
        help="List of encoded CSV files, one per encoder.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=list(DEFAULT_NAMES),
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
        help="Date column used to create the year field.",
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


def add_encoding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-json",
        default="data/IMDB_reviews.json",
        help="Raw JSON/JSONL review dataset to encode.",
    )
    parser.add_argument(
        "--encoded-dir",
        default="data/encoded",
        help="Directory where encoded CSV files will be written.",
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip preprocessing/encoding and use existing encoded CSV files.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Encode only a small subset of rows for testing.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of rows to encode in test mode.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=300_000,
        help="Maximum number of raw reviews to encode.",
    )


def build_config(args: argparse.Namespace) -> DriftConfig:
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
    return config


def maybe_run_encoding(args: argparse.Namespace) -> None:
    if args.skip_encoding:
        print("Skipping preprocessing/encoding and using existing encoded CSV files.")
        return

    preprocess_and_encode(
        json_file=args.raw_json,
        output_dir=args.encoded_dir,
        test_mode=args.test_mode,
        test_size=args.test_size,
        max_reviews=args.max_reviews,
    )

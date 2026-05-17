from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from execution.common import add_encoding_args, add_shared_model_args, build_config, maybe_run_encoding
from utils.comparison import compare_all_embeddings
from utils.data import load_reviews
from utils.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end pipeline: preprocessing, encoding, model execution, "
            "evaluation, and comparison outputs."
        )
    )
    add_encoding_args(parser)
    add_shared_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.names) != len(args.input_files):
        raise ValueError("--names must have the same number of values as --input-files")

    maybe_run_encoding(args)
    config = build_config(args)
    results = []

    for idx, file_path in enumerate(args.input_files):
        pretty_name = args.names[idx]
        df = load_reviews(file_path, date_col=args.date_col)

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

        results.append(
            run_full_pipeline(
                df,
                emb_col="__embedding__",
                pretty_name=pretty_name,
                config=config,
            )
        )

    compare_all_embeddings(results, Path(args.output_dir))

    print("\nDone.")
    print("Generated preprocessing outputs, model outputs, tables, and figures.")
    print(f"- Encoded data: {args.encoded_dir}")
    print(f"- Models/tables/figures: {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.comparison import (
    compare_all_embeddings,
    load_results_from_tables,
    regenerate_per_encoder_figures,
)
from utils.config import DriftConfig
from visualization.plot_theme import apply_paper_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate visualization outputs from existing tables without "
            "re-running preprocessing, encoding, or models."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory containing tables/ and figures/.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["MiniLM", "LaBSE", "DistilBERT"],
        help="Encoder names matching subfolders in outputs/tables/.",
    )
    parser.add_argument("--train-end", type=int, default=2005)
    parser.add_argument("--gap-end", type=int, default=2010)
    parser.add_argument("--std-multiplier", type=float, default=0.5)
    parser.add_argument(
        "--drift-fraction-threshold",
        type=float,
        default=0.05,
        help="Fraction-significant threshold for KS/AD plots.",
    )
    parser.add_argument(
        "--skip-per-encoder",
        action="store_true",
        help="Only regenerate the cross-encoder comparison figures.",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Only regenerate per-encoder figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    apply_paper_theme()

    config = DriftConfig(
        train_end=args.train_end,
        gap_end=args.gap_end,
        threshold_std_multiplier=args.std_multiplier,
        drift_fraction_threshold=args.drift_fraction_threshold,
        output_dir=output_dir,
    )

    print(f"Loading tables from: {output_dir / 'tables'}")
    print(f"Encoders: {args.names}")
    print(f"Train end / Gap end : {args.train_end} / {args.gap_end}")
    print(f"Std multiplier      : {args.std_multiplier}")

    results = load_results_from_tables(
        args.names,
        output_dir,
        train_end=args.train_end,
        gap_end=args.gap_end,
        std_multiplier=args.std_multiplier,
    )

    if not args.skip_per_encoder:
        for result in results:
            ae_thr = result.get("ae_threshold")
            jsd_thr = result.get("jsd_threshold")
            print(
                f"\n[{result['name']}] "
                f"AE threshold = {ae_thr if ae_thr is None else f'{ae_thr:.6f}'}, "
                f"JSD threshold = {jsd_thr if jsd_thr is None else f'{jsd_thr:.6f}'}"
            )
            regenerate_per_encoder_figures(result, output_dir, config)

    if not args.skip_comparisons:
        print("\nGenerating cross-encoder comparison figures...")
        compare_all_embeddings(results, output_dir)

    print("\nDone.")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"Tables : {output_dir / 'tables'}")


if __name__ == "__main__":
    main()

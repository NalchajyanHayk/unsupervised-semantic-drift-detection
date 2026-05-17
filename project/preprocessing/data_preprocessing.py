"""Convenience entry point for the preprocessing and encoding workflow."""

from .encoder_pipeline import load_json_data, preprocess_and_encode

__all__ = ["load_json_data", "preprocess_and_encode"]


def main() -> None:
    """Run the default preprocessing workflow from the reference project."""
    preprocess_and_encode(test_mode=False, max_reviews=300_000)


if __name__ == "__main__":
    main()

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEBUG_LOG_PATH = Path("/Users/hayknalchajyan/Desktop/capstone _project/.cursor/debug-db4726.log")
DEBUG_SESSION_ID = "db4726"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


class ReviewEncoder:
    def __init__(self, model_names: list = None):
        """Initialize encoders for reviews"""
        if model_names is None:
            model_names = ['all-MiniLM-L6-v2', 'distilbert-base-multilingual-cased', 'LaBSE']
        
        # #region agent log
        _debug_log(
            "H1",
            "encoder_pipeline.py:29",
            "ReviewEncoder init start",
            {"model_names": model_names, "model_count": len(model_names)},
        )
        # #endregion
        self.models = {}
        for idx, name in enumerate(model_names):
            # #region agent log
            _debug_log(
                "H1",
                "encoder_pipeline.py:38",
                "Model load start",
                {"model_name": name, "index": idx},
            )
            # #endregion
            self.models[name] = SentenceTransformer(name)
            # #region agent log
            _debug_log(
                "H1",
                "encoder_pipeline.py:46",
                "Model load done",
                {"model_name": name, "index": idx},
            )
            # #endregion
        self.model_names = model_names
    
    def encode_reviews(self, reviews: list) -> dict:
        """Encode reviews using all models"""
        encodings = {}
        total_reviews = len(reviews)
        # #region agent log
        _debug_log(
            "H2",
            "encoder_pipeline.py:57",
            "encode_reviews start",
            {"total_reviews": total_reviews, "model_count": len(self.model_names)},
        )
        # #endregion
        
        for idx, model_name in enumerate(self.model_names, 1):
            print(f"\n[{idx}/{len(self.model_names)}] Encoding with {model_name}...")
            print(f"Processing {total_reviews} reviews...")
            
            model = self.models[model_name]
            # #region agent log
            _debug_log(
                "H2",
                "encoder_pipeline.py:71",
                "Model encode start",
                {"model_name": model_name, "index": idx, "total_reviews": total_reviews},
            )
            # #endregion
            embeddings = model.encode(reviews, show_progress_bar=True)
            # #region agent log
            _debug_log(
                "H2",
                "encoder_pipeline.py:80",
                "Model encode done",
                {
                    "model_name": model_name,
                    "index": idx,
                    "embedding_rows": int(len(embeddings)),
                    "embedding_dim": int(np.array(embeddings).shape[1]),
                },
            )
            # #endregion
            
            encodings[model_name] = embeddings
            print(f"✓ Completed: {total_reviews} reviews encoded with {model_name}")
        
        return encodings


def load_json_data(data_path: str) -> pd.DataFrame:
    """Load JSON data from data/ directory (handles both JSON and JSONL formats)"""
    print(f"Loading data from {data_path}...")
    
    try:
        # Try loading as standard JSON array first
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # If that fails, try loading as JSONL (one JSON object per line)
        print("  Standard JSON failed, trying JSONL format...")
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"✓ Loaded {len(df)} records")
    return df


def preprocess_and_encode(
    json_file: str = "data/IMDB_reviews.json",
    output_dir: str = "data/encoded",
    test_mode: bool = False,
    test_size: int = 100,
    max_reviews: int = 300_000
) -> None:
    """
    Main preprocessing workflow:
    1. Load JSON data
    2. Encode reviews with multiple models
    3. Save encodings to CSV
    """
    
    print("=" * 60)
    print("REVIEW ENCODING PIPELINE")
    if test_mode:
        print(f"[TEST MODE - Processing first {test_size} rows]")
    elif max_reviews is not None:
        print(f"[Processing first {max_reviews} rows]")
    print("=" * 60)
    # #region agent log
    _debug_log(
        "H3",
        "encoder_pipeline.py:132",
        "preprocess_and_encode start",
        {
            "json_file": json_file,
            "output_dir": output_dir,
            "test_mode": test_mode,
            "test_size": test_size,
            "max_reviews": max_reviews,
        },
    )
    # #endregion
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}\n")
    
    # Load data
    df = load_json_data(json_file)
    
    # Limit to test size if in test mode
    if test_mode:
        df = df.head(test_size)
        print(f"✓ Limited to first {test_size} rows for testing\n")
    elif max_reviews is not None:
        df = df.head(max_reviews)
        print(f"✓ Limited to first {len(df)} rows for encoding\n")
    
    # Ensure required columns exist
    if 'review_date' not in df.columns or 'review_text' not in df.columns:
        raise ValueError("JSON must contain 'review_date' and 'review_text' columns")
    
    print(f"✓ Required columns found: 'review_date', 'review_text'\n")
    
    # Initialize encoder
    print("Initializing encoders...")
    encoder = ReviewEncoder()
    print(f"✓ {len(encoder.models)} encoders initialized\n")
    
    # Encode reviews
    print("=" * 60)
    print("ENCODING PHASE")
    print("=" * 60)
    encodings = encoder.encode_reviews(df['review_text'].tolist())
    
    # Save encodings for each model
    print("\n" + "=" * 60)
    print("SAVING PHASE")
    print("=" * 60)
    
    for idx, (model_name, embeddings) in enumerate(encodings.items(), 1):
        df_encoded = df.copy()
        
        # Add embeddings as columns
        embedding_array = np.array(embeddings)
        embedding_dim = embedding_array.shape[1]
        
        print(f"\n[{idx}/{len(encodings)}] Saving {model_name}...")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Total records: {len(df_encoded)}")
        
        for i in range(embedding_dim):
            df_encoded[f'embedding_{i}'] = embedding_array[:, i]
        
        # Save to CSV
        output_file = f"{output_dir}/reviews_encoded_{model_name.replace('/', '_')}.csv"
        df_encoded.to_csv(output_file, index=False)
        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_file} ({file_size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total reviews processed: {len(df)}")
    print(f"Output files saved to: {output_dir}\n")


if __name__ == "__main__":
    preprocess_and_encode(test_mode=False, max_reviews=300_000)

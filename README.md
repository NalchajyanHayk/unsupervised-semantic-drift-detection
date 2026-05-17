# Language in Motion: Semantic Drift Detection

This repository contains the capstone project **Language in Motion: Unsupervised Detection of Semantic Drift in Text Data**. The project studies how the semantic distribution of time-stamped text changes over time without labeled data.

The workflow uses IMDB reviews with dates, encodes them with three sentence encoders, groups them by year, and compares later years against an early historical reference period. The pipeline evaluates semantic drift with:

- autoencoder reconstruction error
- Jensen-Shannon divergence
- Kolmogorov-Smirnov tests
- Anderson-Darling tests
- Sliced Wasserstein Distance in the original embedding space
- Sliced Wasserstein Distance in the autoencoder latent space

## Project Objective

The goal is to build a reproducible unsupervised framework for semantic drift detection in time-stamped text data and compare how stable the detected drift is across multiple embedding models.

The project uses these encoders:

- `all-MiniLM-L6-v2`
- `LaBSE`
- `distilbert-base-multilingual-cased`

## Current Repository Structure

```text
.
├── data/
│   ├── IMDB_reviews.json
│   └── encoded/
│       ├── reviews_encoded_all-MiniLM-L6-v2.csv
│       ├── reviews_encoded_LaBSE.csv
│       └── reviews_encoded_distilbert-base-multilingual-cased.csv
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── paper.pdf
├── project/
│   ├── preprocessing/
│   │   ├── data_preprocessing.py
│   │   ├── encoder_pipeline.py
│   │   └── io_utils.py
│   ├── models/
│   │   ├── models.py
│   │   ├── statistical_tests.py
│   │   └── wasserstein.py
│   ├── utils/
│   │   ├── comparison.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── helpers.py
│   │   ├── pipeline.py
│   │   ├── run_models.py
│   │   └── run_pipeline.py
│   ├── visualization/
│   │   ├── plot_theme.py
│   │   ├── run_visualizations.py
│   │   ├── table_export.py
│   │   └── visualization.py
│   └── execution/
│       ├── common.py
│       ├── run_models.py
│       ├── run_pipeline.py
│       └── run_visualizations.py
├── run_pipeline.sh
├── README.md
└── requirements.txt
```

## Directory Roles

- `project/preprocessing/`: raw data loading, preprocessing, and sentence-encoder generation
- `project/models/`: core statistical and model logic only
- `project/utils/`: shared config, data handling, pipeline orchestration, and comparisons
- `project/visualization/`: plotting, table export, and visualization regeneration
- `project/execution/`: runnable entry points for full pipeline, reduced models, and visualization-only workflows
- `data/`: raw and encoded datasets
- `outputs/`: generated model artifacts, tables, figures, and logs created when the pipeline runs
- `paper/`: paper assets and LaTeX files

## Required Software and Libraries

Recommended Python version:

```text
Python 3.13.5
```

The code requires Python 3.10 or newer.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Main libraries used:

```text
numpy==2.1.2
pandas==2.3.2
matplotlib==3.9.4
scipy==1.16.2
scikit-learn==1.7.2
statsmodels==0.14.6
torch==2.8.0
joblib==1.5.2
sentence-transformers==3.1.1
tqdm==4.67.1
```

## Input Data

The review data comes from the Kaggle IMDB dataset:

```text
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

Place the raw review file at:

```text
data/IMDB_reviews.json
```

The raw data must contain at least:

```text
review_date
review_text
```

The preprocessing code supports standard JSON arrays and JSONL-style files.

## How to Run the Project

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the full pipeline with the shell script

This is the main entry point for the full project:

```bash
./run_pipeline.sh
```

This script will:

1. read `data/IMDB_reviews.json`
2. create or reuse `data/encoded/`
3. encode the first `300000` reviews with the three reference encoders
4. save one encoded CSV per encoder in `data/encoded/`
5. run the full drift-detection and evaluation pipeline
6. regenerate visualization outputs from the generated tables

The expected encoded files are:

```text
data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv
data/encoded/reviews_encoded_LaBSE.csv
data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv
```

The expected output folders are:

```text
outputs/figures/
outputs/tables/
outputs/models/
outputs/logs/
```

### 3. Run the full pipeline directly with Python

If you want to run the full execution layer directly instead of the shell script:

```bash
python project/execution/run_pipeline.py \
  --raw-json data/IMDB_reviews.json \
  --encoded-dir data/encoded \
  --input-files \
    data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv \
    data/encoded/reviews_encoded_LaBSE.csv \
    data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv \
  --names MiniLM LaBSE DistilBERT \
  --output-dir outputs \
  --max-reviews 300000
```

### 4. Run preprocessing and encoding only

You can run the encoder pipeline directly:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "project")

from preprocessing.encoder_pipeline import preprocess_and_encode

preprocess_and_encode(
    json_file="data/IMDB_reviews.json",
    output_dir="data/encoded",
    test_mode=False,
    max_reviews=300000,
)
PY
```

### 5. Run only the reduced model workflow

This runs the autoencoder and Wasserstein-based workflow only:

```bash
python project/execution/run_models.py \
  --raw-json data/IMDB_reviews.json \
  --encoded-dir data/encoded \
  --input-files \
    data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv \
    data/encoded/reviews_encoded_LaBSE.csv \
    data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv \
  --names MiniLM LaBSE DistilBERT \
  --output-dir outputs \
  --max-reviews 300000
```

If encoded files already exist and you want to skip re-encoding:

```bash
python project/execution/run_models.py --skip-encoding
```

### 6. Regenerate visualizations only

If tables already exist and you only want to refresh figures:

```bash
python project/execution/run_visualizations.py \
  --output-dir outputs \
  --names MiniLM LaBSE DistilBERT
```

Useful options:

```bash
python project/execution/run_visualizations.py --skip-comparisons
python project/execution/run_visualizations.py --skip-per-encoder
python project/execution/run_visualizations.py --std-multiplier 0.5
```

## Default Experiment Settings

The default temporal split is:

```text
training/reference period: years <= 2005
gap period:                2006-2010
test/post-gap period:      years > 2010
```

Important default model settings:

```text
autoencoder hidden layers: 512, 384, 256, 128, 96
latent dimension:          64
epochs:                    40
batch size:                64
learning rate:             0.001
AE threshold multiplier:   0.5
SWD projections:           512
SWD bootstrap runs:        200
SWD max samples:           10000 per distribution
random seed:               42
```

## Notes

- `run_pipeline.sh` is the recommended full-project command.
- The encoded CSV files are written directly into `data/encoded/`.
- The visualization regeneration step does not rerun preprocessing or model training; it rebuilds figures from existing output tables.

Per-encoder outputs are written into:

```text
outputs/figures/minilm/
outputs/figures/labse/
outputs/figures/distilbert/
outputs/tables/minilm/
outputs/tables/labse/
outputs/tables/distilbert/
```

Cross-encoder comparison outputs are written into:

```text
outputs/figures/comparisons/
outputs/tables/comparisons/
```

The code saves figures as publication-oriented `.png` and `.pdf` files, and tables as `.csv` files with LaTeX `.tex` siblings for use in the paper.

To regenerate figures and LaTeX table files from already-created CSV tables, without rerunning encoding or model training, use:

```bash
python -u scripts/run_visualizations.py \
  --output-dir outputs \
  --names MiniLM LaBSE DistilBERT \
  --std-multiplier 0.5
```

Use this command when the table CSVs already exist and only the plots or `.tex` table files need to be refreshed.

## Main Output Files

Autoencoder tables:

```text
outputs/tables/<encoder>/<encoder>_ae_all_years.csv
outputs/tables/<encoder>/<encoder>_ae_yearly.csv
outputs/tables/<encoder>/<encoder>_period_error_summary.csv
outputs/tables/<encoder>/<encoder>_ae_training_losses.csv
```

Jensen-Shannon tables:

```text
outputs/tables/<encoder>/<encoder>_jsd_all_years.csv
outputs/tables/<encoder>/<encoder>_jsd_yearly.csv
outputs/tables/<encoder>/<encoder>_period_jsd_summary.csv
```

Statistical-test tables:

```text
outputs/tables/<encoder>/<encoder>_ks.csv
outputs/tables/<encoder>/<encoder>_ad.csv
```

Sliced Wasserstein tables:

```text
outputs/tables/<encoder>/<encoder>_ordinary_swd.csv
outputs/tables/<encoder>/<encoder>_latent_swd.csv
outputs/tables/<encoder>/<encoder>_ordinary_vs_latent_swd.csv
outputs/tables/<encoder>/<encoder>_period_swd.csv
```

Comparison tables:

```text
outputs/tables/comparisons/
```

Trained model files:

```text
outputs/models/<encoder>/autoencoder_state_dict.pt
outputs/models/<encoder>/scaler.joblib
```

## Notes on Runtime

Encoding the raw text with transformer sentence encoders is the slowest preprocessing step. Full model execution can also take time because it trains one autoencoder per encoder and computes bootstrap SWD baselines. The SWD implementation uses `swd_max_samples=10000` by default to keep the bootstrap step tractable on the 300000-review encoded dataset.

## Citation Context

This code accompanies the paper:

```text
Language in Motion: Unsupervised Detection of Semantic Drift in Text Data
Hayk Nalchajyan and Gurgen Hovakimyan
American University of Armenia
```

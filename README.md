# Language in Motion: Unsupervised Detection of Semantic Drift in Text Data

Unsupervised detection of semantic drift in time-stamped text data using text embeddings and distribution-based statistical methods.


This repository contains the code for the capstone project **Language in Motion: Unsupervised Detection of Semantic Drift in Text Data**. The project studies how the semantic distribution of time-stamped text changes over time when labels are not available.

The experiments use IMDB reviews with review dates. Reviews are encoded with three sentence encoders, grouped by year, and compared against an early historical reference period. The pipeline evaluates semantic drift using autoencoder reconstruction error, Jensen-Shannon divergence, dimension-wise Kolmogorov-Smirnov and Anderson-Darling tests, and Sliced Wasserstein Distance in both the original embedding space and the autoencoder latent space.

## Project Objective

The objective is to build a reproducible unsupervised framework for detecting semantic drift in time-stamped text data. The project asks whether semantic changes can be detected without labels, how stable drift signals are across different embedding models, and which unsupervised drift measures are useful at corpus scale.

The paper evaluates three encoders:

- `all-MiniLM-L6-v2`
- `LaBSE`
- `distilbert-base-multilingual-cased`

The main finding reported in the paper is that autoencoder reconstruction error gives the most stable year-by-year signal across encoders, while high-sensitivity statistical tests and bootstrap-based SWD tend to flag all post-gap years at this data scale.

## Repository Structure

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
├── scripts/
│   ├── run_pipeline.py
│   └── run_visualizations.py
├── src/
│   ├── preprocessing/
│   │   └── encoder_pipeline.py
│   └── drift_detection/
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── pipeline.py
│       ├── statistical_tests.py
│       ├── wasserstein.py
│       ├── visualization.py
│       ├── comparison.py
│       ├── plot_theme.py
│       └── table_export.py
├── run_full_pipeline.sh
├── run_models.py
├── run_models.sh
└── requirements.txt
```

## Required Software and Libraries

Recommended Python version:

```text
Python 3.13.5
```

The code requires Python 3.10 or newer. The project was run in a Python 3.13.5 environment.

Required Python libraries:

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

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not pinned yet, install the full set manually:

```bash
python -m pip install numpy==2.1.2 pandas==2.3.2 matplotlib==3.9.4 scipy==1.16.2 scikit-learn==1.7.2 statsmodels==0.14.6 torch==2.8.0 joblib==1.5.2 sentence-transformers==3.1.1 tqdm==4.67.1
```

## Input Data

Place the raw review file at:

```text
data/IMDB_reviews.json
```

The raw data must contain at least these columns:

```text
review_date
review_text
```

The preprocessing code supports both standard JSON arrays and JSONL-style files. The default encoding step processes the first `300000` reviews and writes one encoded CSV per sentence encoder into `data/encoded/`.

## Reproducing the Results

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Encode the raw reviews

Run:

```bash
bash run_full_pipeline.sh
```

This script:

- installs the Python requirements
- creates output directories
- loads `data/IMDB_reviews.json`
- encodes the first `300000` reviews with MiniLM, LaBSE, and DistilBERT
- saves encoded files in `data/encoded/`

The encoded outputs are:

```text
data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv
data/encoded/reviews_encoded_LaBSE.csv
data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv
```

You can also run the encoder directly:

```bash
python - <<'PY'
from src.preprocessing.encoder_pipeline import preprocess_and_encode

preprocess_and_encode(
    json_file="data/IMDB_reviews.json",
    output_dir="data/encoded",
    test_mode=False,
    max_reviews=300000,
)
PY
```

### 3. Run the full drift detection pipeline

Run all methods from the paper:

```bash
python -u scripts/run_pipeline.py \
  --input-files \
  data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv \
  data/encoded/reviews_encoded_LaBSE.csv \
  data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv \
  --names MiniLM LaBSE DistilBERT \
  --output-dir outputs
```

This runs:

- autoencoder reconstruction error
- Jensen-Shannon divergence
- Kolmogorov-Smirnov tests
- Anderson-Darling tests
- ordinary Sliced Wasserstein Distance
- autoencoder latent-space Sliced Wasserstein Distance
- cross-encoder comparisons

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

### 4. Run only autoencoder and Sliced Wasserstein models

For the smaller model-only workflow, run:

```bash
bash run_models.sh
```

or:

```bash
python -u run_models.py \
  --input-files \
  data/encoded/reviews_encoded_all-MiniLM-L6-v2.csv \
  data/encoded/reviews_encoded_LaBSE.csv \
  data/encoded/reviews_encoded_distilbert-base-multilingual-cased.csv \
  --names MiniLM LaBSE DistilBERT \
  --output-dir outputs
```

This runs only:

- autoencoder reconstruction error
- ordinary Sliced Wasserstein Distance
- autoencoder latent-space Sliced Wasserstein Distance

To adjust the autoencoder threshold, change the standard-deviation multiplier:

```bash
python -u run_models.py --threshold-std-multiplier 0.5
```

To reduce or increase SWD runtime, change the sample cap:

```bash
python -u run_models.py --swd-max-samples 10000
```

## How Figures and Tables Were Generated

The full pipeline writes figures and tables automatically under:

```text
outputs/figures/
outputs/tables/
outputs/models/
```

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

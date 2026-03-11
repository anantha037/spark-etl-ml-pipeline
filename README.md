# NYC Yellow Taxi — Big Data ETL & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Spark](https://img.shields.io/badge/Apache_Spark-3.5.1-orange?logo=apachespark)
![CI](https://github.com/anantha037/spark-etl-ml-pipeline/actions/workflows/ci.yml/badge.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end big data pipeline built with Apache Spark on the NYC TLC Yellow Taxi dataset (Jan–Mar 2023, **9.3 million trips**). Covers everything from raw Parquet ingestion to a trained ML model with live predictions — all running in under 3 minutes on a laptop.

---

## What This Project Does

Raw NYC taxi Parquet files go in. Cleaned data, trained models, and predictions come out.

```
Raw Parquet Files (3 months, ~1.5GB)
        │
        ▼
  ┌─────────────┐
  │   EXTRACT   │  Schema normalization, union 9.3M rows
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  TRANSFORM  │  Clean, impute nulls, engineer 10 features
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │    LOAD     │  Parquet warehouse (partitioned) + PostgreSQL
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  ML PREP   │  Remove leakage, encode, scale → feature vectors
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │    TRAIN    │  Random Forest (80% accuracy) + Linear Regression (R²=0.943)
  └──────┬──────┘
         │
         ▼
  1,787,917 predictions with confidence scores
```

---

## Results

| Task | Model | Key Metric |
|------|-------|------------|
| Tip classification (generous tipper?) | Random Forest (tuned) | **80.0% accuracy, AUC 0.789** |
| Fare prediction | Linear Regression | **R² = 0.943, MAE = $1.38** |

**ETL:** 9,384,487 raw rows → 8,934,307 clean rows (4.8% removed) in **79 seconds**  
**Full pipeline:** Raw data → predictions in **3 minutes**

---

## Project Structure

```
spark-etl-ml-project/
├── src/
│   ├── etl/
│   │   ├── extract.py          # Read + normalize 3 Parquet files
│   │   ├── transform.py        # Clean, impute, feature engineer
│   │   └── load.py             # Write Parquet warehouse + PostgreSQL
│   ├── ml/
│   │   ├── data_prep.py        # Remove leakage, build feature pipeline
│   │   ├── train_model.py      # Random Forest + Linear Regression
│   │   └── evaluate_and_tune.py # Confusion matrix, CV grid search
│   └── pipeline/
│       ├── etl_pipeline.py     # ETL orchestration
│       ├── etl_optimizer.py    # Benchmark: pruning, caching, AQE
│       └── full_pipeline.py    # End-to-end: raw data → predictions
├── models/
│   ├── feature_pipeline/       # Fitted Spark ML Pipeline
│   ├── random_forest_classifier/
│   ├── linear_regression_fare/
│   └── rf_tuned_best/          # Tuned model (production)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                    # Place source Parquet files here
│   ├── processed/              # Generated — partitioned by month
│   ├── ml/                     # Generated — train/test splits
│   └── output/                 # Generated — predictions
├── jars/
│   └── postgresql-42.7.3.jar
└── docs/
```

---

## Tech Stack

- **Apache Spark 3.5.1** — distributed data processing engine
- **PySpark** — Python API for Spark
- **Python 3.10** — core language
- **Spark MLlib** — Random Forest, Linear Regression, CrossValidator, Pipelines
- **PostgreSQL** — summary + prediction output tables
- **Apache Parquet** — columnar data warehouse format
- **WSL2 Ubuntu** — Linux environment on Windows

---

## Setup

### Prerequisites

- Python 3.10+
- Java 11 (OpenJDK)
- Apache Spark 3.5.1 installed at `/opt/spark`
- PostgreSQL running locally
- WSL2 Ubuntu (if on Windows) — recommended

### Install

```bash
git clone https://github.com/<your-username>/spark-etl-ml-pipeline.git
cd spark-etl-ml-pipeline

python -m venv venv
source venv/bin/activate
pip install pyspark==3.5.1 numpy pandas
```

### Get the Data

Download from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```bash
mkdir -p data/raw
# Download yellow_tripdata_2023-01.parquet, 02, 03 into data/raw/
```

### PostgreSQL Setup

```bash
sudo -u postgres psql
CREATE DATABASE nyc_taxi;
CREATE USER anantha WITH PASSWORD 'spark123';
GRANT ALL PRIVILEGES ON DATABASE nyc_taxi TO anantha;
\q
```

Update the `POSTGRES_URL` and `POSTGRES_PROPS` in `full_pipeline.py` with your credentials.

---

## Running the Pipeline

### Full Pipeline (recommended)

```bash
python src/pipeline/full_pipeline.py
```

This runs everything: Extract → Transform → Load → ML Prep → Train → Predict → Save.

### Individual Phases

```bash
# ETL only
python src/pipeline/etl_pipeline.py

# Optimization benchmarks
python src/pipeline/etl_optimizer.py

# ML training only (needs ETL done first)
python src/ml/train_model.py

# Evaluation + hyperparameter tuning
python src/ml/evaluate_and_tune.py
```

---

## Key Technical Decisions

**Schema normalization across files**  
The January file encodes some columns as BIGINT while Feb/Mar use INT32. Reading files individually, casting to LongType, then unioning on common columns handles this. Using Spark's built-in `mergeSchema` does not work here because it cannot resolve the type conflict.

**Lazy evaluation + single persist**  
Calling `.count()` after each transform step caused JVM OOM crashes (full 9.3M row scan per step). The fix: `StorageLevel.MEMORY_AND_DISK` after extract, then all transforms stay lazy. Only 2 `.count()` calls in the entire pipeline.

**Removing data leakage**  
`tip_amount` and `tip_pct` directly encode the target (`generous_tipper`). Including them gives 99%+ fake accuracy. Properly removing these plus `total_amount` and `fare_per_mile` brings accuracy to its honest 80%.

**Payment type dominates**  
Feature importance shows `payment_type` at 93.4%. This makes sense — cash passengers literally cannot tip electronically in this dataset. This is a dataset artifact, not a model flaw.

---

## Optimization Benchmarks

| Technique | Without | With | Speedup |
|-----------|---------|------|---------|
| Column Pruning | 37.8s | 6.3s | **6.0x** |
| Partition Pruning | 4.5s | 2.1s | **2.1x** |
| AQE | 7.8s | 5.5s | **1.4x** |
| Caching (3 ops) | 16.2s | 59.4s | 0.3x (SSD faster) |

---

## ML Results Detail

**Random Forest Classifier — Confusion Matrix (test set: 1,787,917 rows)**

|  | Predicted Not Generous | Predicted Generous |
|--|--|--|
| **Actual Not Generous** | 327,423 (TN) | 358,154 (FP) |
| **Actual Generous** | 59 (FN) | 1,102,281 (TP) |

The model almost never misses a real generous tipper (FN = 59 out of 1.1M). The false positives are card-paying passengers whose tip ended up below 20%.

---

## What's Next

- [ ] Streamlit dashboard for interactive trip predictions
- [ ] FastAPI endpoint for real-time inference
- [ ] Docker + docker-compose for one-command setup
- [ ] GitHub Actions CI/CD pipeline
- [ ] Full year (12 months) of data for seasonal modeling

---

## Dataset

NYC TLC Yellow Taxi Trip Records — publicly available at  
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Data files are not included in this repository due to size (~1.5GB). See Setup above to download them.

---

## Author

**Anantha Krishnan**  
Built as a big data engineering and ML portfolio project — March 2026

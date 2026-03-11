# NYC Taxi Big Data Pipeline — Complete Project Documentation

> **Author:** Anantha | **Date:** March 2026 | **Stack:** Apache Spark 3.5.1 · PySpark · FastAPI · Streamlit · Docker · PostgreSQL

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset](#3-dataset)
4. [Environment Setup](#4-environment-setup)
5. [ETL Pipeline](#5-etl-pipeline)
6. [ML Pipeline](#6-ml-pipeline)
7. [Optimization Results](#7-optimization-results)
8. [FastAPI Endpoint](#8-fastapi-endpoint)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Docker Deployment](#10-docker-deployment)
11. [CI/CD — GitHub Actions](#11-cicd--github-actions)
12. [Key Technical Decisions](#12-key-technical-decisions)
13. [Challenges & Fixes](#13-challenges--fixes)
14. [Results Summary](#14-results-summary)
15. [Project Structure](#15-project-structure)

---

## 1. Project Overview

An end-to-end big data pipeline built on **Apache Spark** that processes 9.3 million NYC Yellow Taxi trip records across three months (Jan–Mar 2023). The pipeline covers the full data engineering lifecycle: raw data ingestion → cleaning → feature engineering → ML training → model serving → interactive dashboard → containerized deployment.

### Goals

- Process multi-file Parquet data with heterogeneous schemas using Spark's distributed engine
- Build a production-style ETL pipeline with proper error handling and partitioned output
- Train and tune ML models on 7+ million rows using Spark MLlib
- Serve predictions via a REST API and an interactive dashboard
- Package the entire stack in Docker for one-command deployment

### What This Is Not

This is not a notebook experiment. Every component runs as a standalone Python script. The pipeline is designed to be repeatable, testable, and deployable — the same patterns used in production data engineering teams.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA LAYER                           │
│   NYC TLC Parquet files (Jan / Feb / Mar 2023) — 9.38M rows     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ETL PIPELINE                             │
│  extract.py → transform.py → load.py                           │
│  • Schema normalization (INT32 → BIGINT across months)          │
│  • 17 quality filters applied                                   │
│  • 10 features engineered                                       │
│  • Output: Parquet (partitioned) + PostgreSQL                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE                              │
│  data_prep.py → train_model.py → evaluate_and_tune.py          │
│  • 80/20 train/test split → 7.1M train rows                    │
│  • Random Forest Classifier (generous tipper prediction)        │
│  • Linear Regression (fare amount prediction)                   │
│  • 3-fold CrossValidator, 8-combo ParamGrid                    │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐      ┌───────────────────────┐
│   FastAPI        │      │  Streamlit Dashboard   │
│   REST API       │      │  3-tab interactive UI  │
│   /predict       │      │  Trip Explorer         │
│   /predict/batch │      │  Tip Predictor         │
│   /model/info    │      │  Pipeline Stats        │
└──────────┬───────┘      └───────────┬────────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
           ┌─────────────────────────┐
           │    Docker Compose       │
           │  API + Dashboard + DB   │
           │  docker-compose up      │
           └─────────────────────────┘
```

---

## 3. Dataset

| Property | Value |
|----------|-------|
| Source | NYC Taxi & Limousine Commission (TLC) |
| Format | Parquet (columnar) |
| Period | January – March 2023 |
| Raw rows | 9,384,487 |
| Files | 3 monthly Parquet files + taxi_zone_lookup.csv |
| Raw size | ~500 MB |

### Raw Schema (selected columns)

| Column | Type | Description |
|--------|------|-------------|
| `tpep_pickup_datetime` | timestamp | Pickup timestamp |
| `tpep_dropoff_datetime` | timestamp | Dropoff timestamp |
| `passenger_count` | long | Number of passengers |
| `trip_distance` | double | Miles traveled |
| `RatecodeID` | long | Rate code (1=Standard, 2=JFK, etc.) |
| `PULocationID` | long | Pickup TLC zone ID |
| `DOLocationID` | long | Dropoff TLC zone ID |
| `payment_type` | long | 1=Credit Card, 2=Cash, 3=No Charge, 4=Dispute |
| `fare_amount` | double | Base meter fare (USD) |
| `tip_amount` | double | Tip amount (USD) |
| `total_amount` | double | Total charged (USD) |

### Schema Issue Across Months

A critical discovery: `passenger_count`, `RatecodeID`, and `payment_type` are stored as **INT32** in January but **BIGINT (INT64)** in February and March. Reading all three files into a single DataFrame with `spark.read.parquet("*")` silently fails the union. This required manual schema normalization before combining files.

---

## 4. Environment Setup

### System Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 (WSL2) |
| Python | 3.10 |
| Java | 11+ (required for PySpark) |
| Apache Spark | 3.5.1 |
| PostgreSQL | 15 |
| RAM | 6GB allocated to WSL2 (8GB total) |

### WSL2 Memory Configuration

```ini
# ~/.wslconfig (Windows host)
[wsl2]
memory=6GB
processors=4
```

### Local Setup

```bash
# Clone the repo
git clone https://github.com/anantha037/spark-etl-ml-pipeline
cd spark-etl-ml-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# PostgreSQL setup
sudo -u postgres psql
CREATE DATABASE nyc_taxi;
CREATE USER anantha WITH PASSWORD 'spark123';
GRANT ALL PRIVILEGES ON DATABASE nyc_taxi TO anantha;
```

### Download Raw Data

Download the NYC TLC Yellow Taxi Parquet files for Jan–Mar 2023 from:
`https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`

Place them in `data/raw/`:
```
data/raw/
├── yellow_tripdata_2023-01.parquet
├── yellow_tripdata_2023-02.parquet
└── yellow_tripdata_2023-03.parquet
```

---

## 5. ETL Pipeline

### 5.1 Extract (`src/etl/extract.py`)

Reads each Parquet file individually rather than using a wildcard glob. This is deliberate — a wildcard read applies the schema of the first file to all others, silently corrupting INT32 columns in later files.

```python
def normalize_schema(df):
    """Cast known integer columns to LongType across all monthly files."""
    INT_COLS = ['passenger_count', 'ratecodeid', 'payment_type',
                'pulocationid', 'dolocationid']
    for col_name in INT_COLS:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(LongType()))
    return df
```

After normalization, files are unioned on their common column set. This handles cases where a column exists in some months but not others.

**Output:** Single unified DataFrame, 9,384,487 rows.

### 5.2 Transform (`src/etl/transform.py`)

#### Quality Filters Applied

| Filter | Condition | Rows Removed |
|--------|-----------|-------------|
| Fare amount | `fare_amount >= 2.50` | ~12K |
| Trip distance | `trip_distance > 0` | ~8K |
| Passenger count | `1 <= passenger_count <= 6` | ~85K |
| Tip sanity | `tip_amount >= 0` | ~1K |
| Total amount | `total_amount > 0` | ~2K |
| Datetime validity | dropoff > pickup | ~5K |
| Duration | `1 min <= duration <= 180 min` | ~180K |
| Speed | avg speed <= 100 mph | ~30K |
| Payment-tip consistency | cash tips = 0 | ~127K |
| **Total removed** | | **450,180 (4.8%)** |

#### Features Engineered

| Feature | Description |
|---------|-------------|
| `trip_duration_min` | `(dropoff - pickup) / 60` |
| `pickup_hour` | Hour of day (0–23) |
| `pickup_dayofweek` | Day of week (1=Sunday, 7=Saturday) |
| `is_weekend` | 1 if Saturday or Sunday |
| `time_of_day` | morning / afternoon / evening / night |
| `is_airport_trip` | 1 if zone ID is JFK (132), LGA (138), or EWR (1) |
| `avg_speed_mph` | `distance / (duration / 60)` |
| `tip_pct` | `tip_amount / fare_amount * 100` |
| `generous_tipper` | Binary label: 1 if `tip_pct > 20` |
| `source_month` | Partition key: "2023-01", "2023-02", "2023-03" |

### 5.3 Load (`src/etl/load.py`)

Two write destinations:

**Parquet (primary):**
```python
df.write \
  .mode("overwrite") \
  .partitionBy("source_month") \
  .parquet("data/processed/")
```

**PostgreSQL (secondary):**
```python
df.write \
  .format("jdbc") \
  .option("url", "jdbc:postgresql://localhost:5432/nyc_taxi") \
  .option("dbtable", "trips_clean") \
  .option("driver", "org.postgresql.Driver") \
  .mode("overwrite") \
  .save()
```

### 5.4 ETL Results

| Month | Raw Rows | Clean Rows | Removed |
|-------|----------|------------|---------|
| 2023-01 | 3,066,766 | 2,925,717 | 141,049 |
| 2023-02 | 2,913,955 | 2,770,759 | 143,196 |
| 2023-03 | 3,403,766 | 3,237,831 | 165,935 |
| **Total** | **9,384,487** | **8,934,307** | **450,180** |

**ETL Runtime: 79 seconds**

---

## 6. ML Pipeline

### 6.1 Data Preparation (`src/ml/data_prep.py`)

- Load from `data/processed/` (post-ETL Parquet)
- Drop leakage columns: `tip_amount`, `tip_pct`, `total_amount`
- 80/20 stratified split → 7,146,390 train / 1,787,917 test
- Feature vector assembled using `VectorAssembler`

#### Feature Vector (26 dimensions)

```
payment_type, trip_distance, fare_amount, trip_duration_min,
pickup_hour, pickup_dayofweek, is_weekend, is_airport_trip,
passenger_count, ratecodeid, pulocationid, dolocationid,
congestion_surcharge, mta_tax, improvement_surcharge,
extra, tolls_amount, avg_speed_mph, store_and_fwd_flag_encoded,
time_of_day_encoded (one-hot)
```

### 6.2 Random Forest Classifier (`src/ml/train_model.py`)

**Task:** Binary classification — predict `generous_tipper` (tip > 20% of fare)

**Pipeline:**
```
StringIndexer → OneHotEncoder → VectorAssembler → RandomForestClassifier
```

**Base model parameters:**
```python
RandomForestClassifier(
    labelCol="generous_tipper",
    featuresCol="features",
    numTrees=20,
    maxDepth=5,
    seed=42
)
```

### 6.3 Hyperparameter Tuning (`src/ml/evaluate_and_tune.py`)

```python
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .addGrid(rf.minInstancesPerNode, [1, 5]) \
    .build()  # 8 combinations

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
    numFolds=3  # 24 total fits on 7.1M rows
)
```

**Best parameters:** `numTrees=50`, `maxDepth=10`, `minInstancesPerNode=1`

### 6.4 Classifier Results

| Metric | Value |
|--------|-------|
| Accuracy | **80.0%** |
| AUC-ROC | **0.789** |
| F1 Score | **0.778** |
| Precision | **0.849** |
| Recall | **0.800** |

**Confusion Matrix:**

| | Predicted: Not Generous | Predicted: Generous |
|---|---|---|
| **Actual: Not Generous** | 327,423 (TN) | 358,154 (FP) |
| **Actual: Generous** | 59 (FN) | 1,102,281 (TP) |

> FN = 59 out of 1.1M genuine generous tippers. Near-perfect recall for class 1.

**Feature Importance:**

| Feature | Importance |
|---------|-----------|
| payment_type | **93.4%** |
| fare_amount | 2.1% |
| congestion_surcharge | 1.8% |
| trip_distance | 1.2% |
| ratecodeid | 0.8% |

Payment type dominates because cash passengers cannot tip electronically. The model learned this real-world constraint directly from data.

### 6.5 Linear Regression (Fare Prediction)

**Task:** Predict `fare_amount` from trip features (excluding fare from the feature vector)

| Metric | Value |
|--------|-------|
| R² Score | **0.943** |
| RMSE | **$4.04** |
| MAE | **$1.38** |
| Training Time | 59.8 seconds |

> The model explains 94.3% of fare variance. Trip distance is the strongest predictor.

---

## 7. Optimization Results

All benchmarks run on a single laptop (8GB RAM, WSL2, SSD) using `src/pipeline/etl_optimizer.py`.

| Technique | Without | With | Speedup | Why It Works |
|-----------|---------|------|---------|--------------|
| Column Pruning | 37.8s | 6.3s | **6.0x** | Parquet is columnar — unused columns are never read from disk |
| Partition Pruning | 4.5s | 2.1s | **2.1x** | Spark reads only the `source_month=2023-01` folder instead of all three |
| AQE (Adaptive Query Execution) | 7.8s | 5.5s | **1.4x** | Runtime statistics allow better join/shuffle plans |
| Caching (3 operations) | 16.2s | 59.4s | **0.3x** | SSD is faster than the JVM for 3 operations; caching overhead exceeds savings |

### Key Insight on Caching

Caching is not always faster. The break-even point is roughly 8–10 operations on the same DataFrame. With only 3 operations, the cost of serializing 8.9M rows into JVM memory exceeded the read time from a fast SSD. In a production environment with HDDs or remote storage (S3, HDFS), caching would have shown a speedup even for 3 operations.

---

## 8. FastAPI Endpoint

**File:** `api.py` | **Port:** 8000

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Welcome + endpoint directory |
| GET | `/health` | Health check — model availability, API version |
| GET | `/model/info` | Model metadata, training stats, metrics |
| GET | `/stats` | Full pipeline statistics (ETL + ML) |
| POST | `/predict` | Single trip prediction |
| POST | `/predict/batch` | Batch predictions (up to 100 trips) |

### Prediction Request Schema

```json
{
  "payment_type": 1,
  "trip_distance": 3.5,
  "fare_amount": 15.0,
  "trip_duration_min": 20.0,
  "pickup_hour": 18,
  "pickup_dayofweek": 5,
  "passenger_count": 1,
  "is_airport_trip": 0,
  "ratecodeid": 1
}
```

### Prediction Response Schema

```json
{
  "prediction": 1,
  "prediction_label": "Generous Tipper",
  "confidence": 0.76,
  "confidence_pct": "76.0%",
  "inference_method": "rule_based_fallback",
  "inference_time_ms": 9144.5,
  "input_summary": {
    "payment_type": "Credit Card",
    "trip_distance_mi": 3.5,
    "fare_amount_usd": 15.0,
    "time_of_day": "evening",
    "is_airport_trip": false
  }
}
```

### Inference Strategy

The API uses a two-tier fallback:

1. **Spark model** — loads `models/rf_tuned_best/` and runs real Random Forest inference if model files exist
2. **Rule-based fallback** — if Spark is unavailable (e.g., container memory constraints), falls back to a rule-based predictor that mirrors the model's learned behavior (cash → 0.06 confidence, credit card → 0.72+ base with adjustments)

### Input Validation

All inputs are validated with Pydantic. Invalid requests return `HTTP 422` with descriptive error messages. A cross-field validator checks that average speed (`distance / duration`) does not exceed 100 mph.

### Run Locally

```bash
source venv/bin/activate
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 9. Streamlit Dashboard

**File:** `dashboard.py` | **Port:** 8501

### Tab 1 — Trip Explorer

- Loads 50K sampled trips from processed Parquet using PyArrow (no Spark required)
- Filters: month, time of day, trip type, payment type
- KPI cards: avg fare, avg tip, avg distance, avg duration, % generous tippers
- Charts: trips by hour, fare distribution, avg fare by time of day, generous tippers by payment type

### Tab 2 — Tip Predictor

- Input controls for all trip features (sliders + dropdowns)
- Calls `try_spark_predict()` → falls back to `fallback_predict()` if Spark unavailable
- Shows: prediction label, confidence score, trip summary table, feature importance chart

### Tab 3 — Pipeline Stats

- ETL metrics: raw vs clean row counts, monthly breakdown, runtime
- ML metrics: accuracy, AUC, F1, confusion matrix display
- Linear regression metrics: R², RMSE, MAE
- Optimization benchmark bar chart
- Full pipeline timing breakdown

### Run Locally

```bash
source venv/bin/activate
streamlit run dashboard.py
# http://localhost:8501
```

---

## 10. Docker Deployment

The full stack (API + Dashboard + PostgreSQL) is containerized using Docker Compose. Anyone with Docker installed can clone this repo and run everything with one command.

### Services

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `postgres` | postgres:15-alpine | 5432 | PostgreSQL database |
| `api` | Built from Dockerfile.api | 8000 | FastAPI prediction endpoint |
| `dashboard` | Built from Dockerfile.dashboard | 8501 | Streamlit dashboard |

### Quick Start

```bash
git clone https://github.com/anantha037/spark-etl-ml-pipeline
cd spark-etl-ml-pipeline

# Place your data files in data/raw/ first
docker-compose up --build
```

First build: ~10 minutes (downloads Java 21 + Python packages)
Subsequent starts: ~30 seconds

**Access:**
- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Architecture

```
docker-compose up
    │
    ├── nyc_taxi_postgres  (starts first, healthcheck on pg_isready)
    │       ↓
    ├── nyc_taxi_api       (waits for postgres healthy, then starts uvicorn)
    │       ↓
    └── nyc_taxi_dashboard (waits for api healthy, then starts streamlit)
```

Health checks are configured on all three services. The dashboard will not start until the API passes its health check, ensuring a clean startup sequence.

### Volumes

Processed data and models are mounted read-only into both containers:

```yaml
volumes:
  - ./data/processed:/app/data/processed:ro
  - ./models:/app/models:ro
```

This keeps the images small (~1.5GB) — raw and processed data stay on the host.

---

## 11. CI/CD — GitHub Actions

**File:** `.github/workflows/ci.yml`

Every push to `main` triggers an automated test suite that:

1. Starts the FastAPI server on the GitHub Actions runner
2. Tests all endpoints with assertions on response structure and values
3. Verifies model metrics are above expected thresholds
4. Confirms invalid inputs return `HTTP 422`

### Tests Run

| Test | Assertion |
|------|-----------|
| `GET /health` | `status == "healthy"`, `api_version` present |
| `GET /model/info` | `accuracy > 0.75`, `auc_roc > 0.70`, row counts correct |
| `GET /stats` | ETL and ML counts match expected values |
| `POST /predict` (credit card) | Returns 0 or 1, confidence in range, all fields present |
| `POST /predict` (cash) | `prediction == 0`, `confidence < 0.3` |
| `POST /predict/batch` | Returns exactly 2 predictions for 2 inputs |
| Validation test | Invalid input returns `HTTP 422` |

**Runtime:** ~33 seconds per run

---

## 12. Key Technical Decisions

### Why PySpark Instead of Pandas?

Pandas would load all 9.3M rows into RAM at once — roughly 4–5GB for this schema. On an 8GB laptop running WSL2 + VS Code, that leaves no room for the JVM. Spark's lazy evaluation and distributed execution model allow processing larger-than-memory datasets by streaming data through the pipeline in stages.

### Why Parquet Instead of CSV?

Parquet is a columnar format. When you run `SELECT fare_amount, trip_distance FROM trips`, Spark reads only those two columns from disk. For a schema with 20+ columns, this means 10–18x less I/O than CSV. Column pruning delivered a **6x speedup** in benchmarks. Parquet also stores schema and statistics (min/max per column) that enable partition pruning without scanning data.

### Why Partition by `source_month`?

Queries that filter by month (e.g., "show January trips") only scan the `source_month=2023-01/` directory. Without partitioning, Spark scans all three months and filters in memory — 3x more I/O. The partition pruning benchmark showed a **2.1x speedup** for single-month queries.

### Why Not Cache Everything?

Caching was **slower** in benchmarks (0.3x). Spark's caching serializes the entire DataFrame into JVM heap memory. The cost of doing that serialization exceeds the benefit if you only reuse the data 3 times. On SSDs or fast local disks, re-reading from Parquet is cheaper than maintaining the JVM cache. Caching pays off with 8+ operations or when reading from remote/slow storage.

### Why a Two-Tier Inference Strategy?

Starting a full SparkSession inside an API container takes 8–12 seconds for the JVM to initialize. For an API that needs sub-second response times at scale, this would be unacceptable. The rule-based fallback delivers instant responses for the most common case (the payment_type rule captures 93% of the model's behavior) while the real Spark model remains available for complete accuracy when the JVM is already warm.

---

## 13. Challenges & Fixes

### Challenge 1: INT32 vs INT64 Schema Mismatch

**Problem:** Reading all three Parquet files with `spark.read.parquet("data/raw/*.parquet")` silently failed the union — January used INT32 for integer columns, February/March used INT64.

**Fix:** Read each file individually, apply `normalize_schema()` to cast all integer columns to `LongType`, find the intersection of column names across all files, then union on common columns only.

### Challenge 2: JVM Out of Memory Crashes

**Problem:** PySpark was crashing mid-pipeline with `java.lang.OutOfMemoryError`. Root cause: calling `.count()` multiple times on the same DataFrame forced repeated full scans of 9.3M rows, each time creating a new execution plan that materialized the data fresh.

**Fix:**
1. Configure WSL2 to 6GB RAM via `~/.wslconfig`
2. Use `persist(StorageLevel.MEMORY_AND_DISK)` instead of `cache()` to spill to disk when heap fills
3. Eliminate redundant `.count()` calls — warm the DataFrame with one count, then keep all transforms lazy

### Challenge 3: PostgreSQL Authentication

**Problem:** `ALTER USER anantha WITH PASSWORD` failed silently — the PostgreSQL `pg_hba.conf` was set to `trust` for local connections, meaning passwords were not actually validated.

**Fix:** Set the password correctly then verify with `psql -U anantha -W nyc_taxi` to confirm password authentication is working before running the pipeline.

### Challenge 4: Probability Vector to Double Conversion

**Problem:** After Random Forest prediction, the `probability` column is a `Vector` type. Calling `.avg("probability")` throws an error — Spark cannot average a Vector column.

**Fix:** Use `vector_to_array(col("probability"))[1]` to extract the class-1 probability as a `Double` column:
```python
from pyspark.ml.functions import vector_to_array
df = df.withColumn("confidence", vector_to_array(col("probability"))[1])
```

### Challenge 5: JVM Crash After PostgreSQL Write

**Problem:** In the final save stage, calling `summary.show()` after writing predictions to PostgreSQL triggered a full recomputation of the entire prediction pipeline on an already-exhausted JVM.

**Fix:** Collect the summary to a Python dictionary first, then write and display:
```python
summary_row = summary.collect()[0].asDict()  # Materialize before write
# Now write to PostgreSQL — no recomputation triggered
predictions.write.jdbc(...)
# Display from Python dict — no Spark involved
print(f"Total predictions: {summary_row['count']}")
```

---

## 14. Results Summary

### ETL

| Metric | Value |
|--------|-------|
| Raw rows | 9,384,487 |
| Clean rows | 8,934,307 |
| Rows removed | 450,180 (4.8%) |
| Features engineered | 10 |
| Runtime | **79 seconds** |

### ML — Random Forest (Tip Classification)

| Metric | Value |
|--------|-------|
| Training rows | 7,146,390 |
| Test rows | 1,787,917 |
| Accuracy | **80.0%** |
| AUC-ROC | **0.789** |
| F1 Score | **0.778** |
| Precision | **0.849** |
| Recall | **0.800** |

### ML — Linear Regression (Fare Prediction)

| Metric | Value |
|--------|-------|
| R² Score | **0.943** |
| RMSE | **$4.04** |
| MAE | **$1.38** |

### Full Pipeline (End-to-End)

| Stage | Time |
|-------|------|
| ETL | 79.6s |
| ML Prep | 9.2s |
| Predict | 33.5s |
| Save | 54.5s |
| **Total** | **181.5s (3.0 min)** |

Raw Parquet → 1,787,917 predictions in **3 minutes** on a laptop.

---

## 15. Project Structure

```
spark-etl-ml-pipeline/
│
├── src/
│   ├── etl/
│   │   ├── extract.py          # Load + normalize Parquet files
│   │   ├── transform.py        # Clean + feature engineer
│   │   └── load.py             # Write Parquet + PostgreSQL
│   ├── ml/
│   │   ├── data_prep.py        # Train/test split + feature vectors
│   │   ├── train_model.py      # RF classifier + LR regressor
│   │   └── evaluate_and_tune.py# CrossValidator + metrics
│   └── pipeline/
│       ├── etl_pipeline.py     # Orchestrate ETL phases
│       ├── etl_optimizer.py    # Benchmark optimization techniques
│       └── full_pipeline.py    # End-to-end ETL → ML → predictions
│
├── models/
│   ├── feature_pipeline/       # Saved PipelineModel (indexers + assembler)
│   ├── random_forest_classifier/  # Base RF model
│   ├── rf_tuned_best/          # Tuned RF model (used for serving)
│   └── linear_regression_fare/ # Fare prediction model
│
├── data/
│   ├── raw/                    # Source Parquet files (not in git)
│   ├── processed/              # Cleaned Parquet (partitioned by source_month)
│   ├── ml/                     # train/ and test/ splits
│   └── output/                 # predictions/ and summary/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions — 7 API tests on every push
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── api.py                      # FastAPI REST endpoint
├── dashboard.py                # Streamlit interactive dashboard
├── Dockerfile.api              # API container (Python 3.10 + Java 21)
├── Dockerfile.dashboard        # Dashboard container
├── docker-compose.yml          # API + Dashboard + PostgreSQL
├── requirements.txt            # Python dependencies
└── README.md
```

---

*Built by Anantha — March 2026*

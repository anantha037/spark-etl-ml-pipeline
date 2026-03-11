# NYC Taxi Big Data Pipeline

![CI](https://github.com/anantha037/spark-etl-ml-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end big data pipeline on **9.3 million NYC Yellow Taxi trip records** — ETL, ML, REST API, interactive dashboard, and Docker deployment.

---

## What This Project Does

Raw Parquet files → Spark ETL → ML training → FastAPI predictions → Streamlit dashboard → Docker

The pipeline processes NYC taxi data at scale, trains a Random Forest model to predict generous tippers, and serves predictions through a REST API and interactive dashboard — all packaged in Docker for one-command deployment.

---

## Quick Start (Docker)

```bash
git clone https://github.com/anantha037/spark-etl-ml-pipeline
cd spark-etl-ml-pipeline

# Add raw data files to data/raw/ (see Dataset section)
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| API Health Check | http://localhost:8000/health |

First build takes ~10 minutes (Java 21 + Python packages). Subsequent starts take ~30 seconds.

---

## Results

### ETL Pipeline

| Metric | Value |
|--------|-------|
| Raw rows | 9,384,487 |
| Clean rows | 8,934,307 |
| Rows removed | 450,180 (4.8%) |
| Runtime | **79 seconds** |

### ML — Random Forest (Tip Classification)

| Metric | Value |
|--------|-------|
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

### Full Pipeline Runtime

| Stage | Time |
|-------|------|
| ETL | 79.6s |
| ML Prep | 9.2s |
| Prediction | 33.5s |
| Save | 54.5s |
| **Total** | **~3 minutes** |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              RAW DATA (9.38M rows)                  │
│   3x Monthly Parquet files — NYC TLC Yellow Taxi    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                  ETL PIPELINE                       │
│  extract.py → transform.py → load.py               │
│  • Schema normalization across months               │
│  • 17 quality filters → 450K rows removed          │
│  • 10 features engineered                          │
│  • Output: partitioned Parquet + PostgreSQL         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                  ML PIPELINE                        │
│  data_prep → train_model → evaluate_and_tune       │
│  • 80/20 split → 7.1M train rows                  │
│  • Random Forest (classification)                  │
│  • Linear Regression (fare prediction)             │
│  • 3-fold CV, 8-combo hyperparameter tuning        │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────┐
│   FastAPI        │   │  Streamlit Dashboard  │
│   api.py         │   │  dashboard.py         │
│   :8000          │   │  :8501                │
└──────────┬───────┘   └──────────┬────────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │    Docker Compose      │
         │  API + Dashboard + DB  │
         └────────────────────────┘
```

---

## Project Structure

```
spark-etl-ml-pipeline/
├── src/
│   ├── etl/
│   │   ├── extract.py          # Load + normalize Parquet files
│   │   ├── transform.py        # Clean + feature engineer
│   │   └── load.py             # Write Parquet + PostgreSQL
│   ├── ml/
│   │   ├── data_prep.py        # Train/test split, feature vectors
│   │   ├── train_model.py      # RF classifier + LR regressor
│   │   └── evaluate_and_tune.py# CrossValidator + metrics
│   └── pipeline/
│       ├── etl_pipeline.py     # Orchestrate ETL
│       ├── etl_optimizer.py    # Benchmark optimizations
│       └── full_pipeline.py    # End-to-end runner
│
├── models/
│   ├── feature_pipeline/       # Saved PipelineModel
│   ├── rf_tuned_best/          # Tuned RF model (served by API)
│   └── linear_regression_fare/ # Fare prediction model
│
├── .github/workflows/
│   └── ci.yml                  # Automated API tests on every push
│
├── api.py                      # FastAPI REST endpoint
├── dashboard.py                # Streamlit dashboard
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── docs/
    └── PROJECT_DOCUMENTATION.md
```

---

## Dataset

NYC TLC Yellow Taxi Trip Records — January, February, March 2023.

Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Place files in `data/raw/`:
```
data/raw/
├── yellow_tripdata_2023-01.parquet
├── yellow_tripdata_2023-02.parquet
└── yellow_tripdata_2023-03.parquet
```

> Raw data files are not included in the repo (too large for git). The `.gitignore` excludes them.

---

## Local Setup (Without Docker)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set up PostgreSQL
sudo -u postgres psql -c "CREATE DATABASE nyc_taxi;"
sudo -u postgres psql -c "CREATE USER anantha WITH PASSWORD 'spark123';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE nyc_taxi TO anantha;"

# 3. Run the full pipeline
python src/pipeline/full_pipeline.py

# 4. Start the API
uvicorn api:app --reload --port 8000

# 5. Start the dashboard
streamlit run dashboard.py
```

---

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "payment_type": 1,
    "trip_distance": 3.5,
    "fare_amount": 15.0,
    "trip_duration_min": 20.0,
    "pickup_hour": 18,
    "pickup_dayofweek": 5
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Generous Tipper",
  "confidence": 0.76,
  "confidence_pct": "76.0%",
  "inference_method": "rule_based_fallback",
  "inference_time_ms": 9.4
}
```

### Other Endpoints

```bash
GET  /health         # Health check
GET  /model/info     # Model metadata and metrics
GET  /stats          # Full pipeline statistics
POST /predict/batch  # Up to 100 trips in one request
```

Full interactive docs at `http://localhost:8000/docs`

---

## Key Technical Decisions

**Why read files individually instead of using a wildcard glob?**
January uses INT32 for integer columns; February and March use INT64. A wildcard read silently applies January's schema to all files, corrupting the union. Reading individually and normalizing with `LongType` casts before unioning fixes this.

**Why `persist(MEMORY_AND_DISK)` instead of `cache()`?**
On an 8GB machine, caching 8.9M rows into JVM heap runs out of memory. `MEMORY_AND_DISK` spills to disk when heap fills, preventing JVM crashes at the cost of slightly slower access on overflow.

**Why is caching shown as 0.3x in benchmarks?**
Caching was slower than re-reading from Parquet for only 3 operations on an SSD. The serialization overhead exceeds savings until you reuse the same DataFrame 8+ times, or when reading from slow remote storage.

**Why a rule-based fallback in the API?**
Starting a SparkSession takes 8–12 seconds for JVM initialization. The rule-based fallback captures 93% of the model's behavior (payment type dominates feature importance) and delivers instant responses when Spark is not warm.

---

## Optimization Benchmarks

| Technique | Without | With | Speedup |
|-----------|---------|------|---------|
| Column Pruning | 37.8s | 6.3s | **6.0x** |
| Partition Pruning | 4.5s | 2.1s | **2.1x** |
| AQE | 7.8s | 5.5s | **1.4x** |
| Caching (3 ops) | 16.2s | 59.4s | 0.3x ⚠️ |

Column pruning wins because Parquet is columnar — unused columns are never read from disk. Caching was slower here because data is on SSD and only 3 operations ran over the cached data.

---

## CI/CD

GitHub Actions runs 7 automated tests on every push to `main`:

- `GET /health` — status and version checks
- `GET /model/info` — accuracy and AUC threshold assertions
- `GET /stats` — ETL and ML row count validation
- `POST /predict` — credit card trip (expects generous prediction)
- `POST /predict` — cash trip (expects not-generous, low confidence)
- `POST /predict/batch` — 2-trip batch
- Validation test — invalid input must return `HTTP 422`

**Runtime: ~33 seconds per run**

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Apache Spark 3.5.1, PySpark |
| Storage | Parquet (partitioned), PostgreSQL 15 |
| ML | Spark MLlib (Random Forest, Linear Regression) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, PyArrow, Plotly |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Language | Python 3.10 |

---

## Full Documentation

See [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for complete technical documentation including schema details, all challenge/fix writeups, feature importance analysis, and optimization deep-dives.

---

*Built by Anantha — March 2026*

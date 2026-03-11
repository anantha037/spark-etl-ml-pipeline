"""
api.py
Phase 14 — FastAPI Prediction Endpoint
NYC Taxi Big Data Pipeline — REST API

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              → Welcome
    GET  /health        → Health check
    GET  /model/info    → Model metadata
    GET  /stats         → Pipeline statistics
    POST /predict       → Single trip prediction
    POST /predict/batch → Batch predictions (up to 100 trips)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import time
import os
import warnings
warnings.filterwarnings("ignore")

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Tip Predictor API",
    description=(
        "REST API for predicting generous tippers in NYC yellow taxi trips. "
        "Built on a Random Forest model trained on 9.3M trip records using Apache Spark MLlib."
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc"       # ReDoc UI
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH   = os.path.join(BASE, "models", "rf_tuned_best")
PIPELINE_PATH = os.path.join(BASE, "models", "feature_pipeline")

# ── Request / Response Schemas ────────────────────────────────────────────────
class TripFeatures(BaseModel):
    """Input features for a single taxi trip prediction."""

    payment_type: int = Field(
        ..., ge=1, le=6,
        description="1=Credit Card, 2=Cash, 3=No Charge, 4=Dispute, 5=Unknown, 6=Voided"
    )
    trip_distance: float = Field(
        ..., gt=0, le=100,
        description="Trip distance in miles"
    )
    fare_amount: float = Field(
        ..., ge=2.5, le=500,
        description="Base meter fare in USD"
    )
    trip_duration_min: float = Field(
        ..., gt=0, le=180,
        description="Trip duration in minutes"
    )
    pickup_hour: int = Field(
        ..., ge=0, le=23,
        description="Hour of pickup (0-23)"
    )
    pickup_dayofweek: int = Field(
        ..., ge=1, le=7,
        description="Day of week: 1=Sunday, 7=Saturday"
    )
    passenger_count: int = Field(
        default=1, ge=1, le=6,
        description="Number of passengers"
    )
    is_airport_trip: int = Field(
        default=0, ge=0, le=1,
        description="1 if pickup or dropoff at JFK, LGA, or EWR"
    )
    ratecodeid: int = Field(
        default=1, ge=1, le=6,
        description="1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated, 6=Group"
    )
    pulocationid: int = Field(
        default=161, ge=1, le=265,
        description="TLC pickup zone ID (1-265)"
    )
    dolocationid: int = Field(
        default=132, ge=1, le=265,
        description="TLC dropoff zone ID (1-265)"
    )
    congestion_surcharge: float = Field(
        default=2.5, ge=0,
        description="NYC congestion surcharge (default $2.50)"
    )

    @validator("trip_duration_min")
    def duration_reasonable(cls, v, values):
        # Cross-field: avg speed should not exceed 100mph
        if "trip_distance" in values and v > 0:
            speed = values["trip_distance"] / (v / 60)
            if speed > 100:
                raise ValueError(
                    f"Average speed {speed:.0f} mph is unrealistic. "
                    "Check trip_distance and trip_duration_min."
                )
        return v

    class Config:
        schema_extra = {
            "example": {
                "payment_type": 1,
                "trip_distance": 3.5,
                "fare_amount": 15.0,
                "trip_duration_min": 20.0,
                "pickup_hour": 18,
                "pickup_dayofweek": 5,
                "passenger_count": 1,
                "is_airport_trip": 0,
                "ratecodeid": 1,
                "pulocationid": 161,
                "dolocationid": 132,
                "congestion_surcharge": 2.5
            }
        }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    prediction: int             = Field(..., description="0=Not generous, 1=Generous tipper")
    prediction_label: str       = Field(..., description="Human-readable label")
    confidence: float           = Field(..., description="Model confidence (0.0–1.0)")
    confidence_pct: str         = Field(..., description="Confidence as percentage string")
    inference_method: str       = Field(..., description="spark_model or rule_based_fallback")
    inference_time_ms: float    = Field(..., description="Inference latency in milliseconds")
    input_summary: dict         = Field(..., description="Echo of key input features")


class BatchPredictionRequest(BaseModel):
    trips: List[TripFeatures] = Field(..., max_items=100, description="Up to 100 trips")


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: List[PredictionResponse]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    spark_model_available: bool
    models_path_exists: bool
    pipeline_path_exists: bool
    api_version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    algorithm: str
    task: str
    training_rows: int
    num_trees: int
    max_depth: int
    feature_vector_size: int
    accuracy: float
    auc_roc: float
    f1_score: float
    precision: float
    recall: float
    top_feature: str
    top_feature_importance: float
    tuning_method: str
    model_path: str


# ── Inference Logic ───────────────────────────────────────────────────────────
def get_time_of_day(hour: int) -> str:
    if hour < 6:   return "night"
    if hour < 12:  return "morning"
    if hour < 18:  return "afternoon"
    return "evening"


def spark_predict(trip: TripFeatures):
    """Attempt real Spark ML inference. Returns (prediction, confidence)."""
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    import pyspark.sql.functions as F

    spark = SparkSession.builder \
        .appName("TaxiAPI") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    row = {
        "trip_distance":       float(trip.trip_distance),
        "fare_amount":         float(trip.fare_amount),
        "trip_duration_min":   float(trip.trip_duration_min),
        "pickup_hour":         int(trip.pickup_hour),
        "pickup_dayofweek":    int(trip.pickup_dayofweek),
        "is_weekend":          int(trip.pickup_dayofweek in [1, 7]),
        "is_airport_trip":     int(trip.is_airport_trip),
        "passenger_count":     int(trip.passenger_count),
        "payment_type":        int(trip.payment_type),
        "pulocationid":        int(trip.pulocationid),
        "dolocationid":        int(trip.dolocationid),
        "avg_speed_mph":       float(trip.trip_distance / (trip.trip_duration_min / 60)),
        "ratecodeid":          int(trip.ratecodeid),
        "congestion_surcharge":float(trip.congestion_surcharge),
        "mta_tax":             0.5,
        "improvement_surcharge": 1.0,
        "extra":               0.5,
        "tolls_amount":        0.0,
        "store_and_fwd_flag":  "N",
        "time_of_day":         get_time_of_day(trip.pickup_hour),
        "generous_tipper":     0,
    }

    df = spark.createDataFrame([row])
    pipeline = PipelineModel.load(PIPELINE_PATH)
    df_feat  = pipeline.transform(df).select("features", "generous_tipper")
    model    = RandomForestClassificationModel.load(MODELS_PATH)
    result   = model.transform(df_feat) \
                    .withColumn("conf", vector_to_array(F.col("probability"))[1]) \
                    .select("prediction", "conf") \
                    .collect()[0]

    return int(result["prediction"]), float(result["conf"])


def rule_based_predict(trip: TripFeatures):
    """Fallback rule-based predictor based on feature importance."""
    payment  = trip.payment_type
    distance = trip.trip_distance
    fare     = trip.fare_amount
    airport  = trip.is_airport_trip
    hour     = trip.pickup_hour

    if payment == 2:                          # Cash — never tips electronically
        return 0, 0.06
    elif payment == 1:                        # Credit card
        base = 0.72
        if airport:           base += 0.08   # Airport tips better
        if hour >= 18:        base += 0.04   # Evening tips better
        if distance > 5:      base += 0.03   # Longer trips
        if fare > 20:         base += 0.03   # Higher fares
        confidence = min(round(base, 4), 0.97)
        return (1 if confidence >= 0.5 else 0), confidence
    else:                                     # No charge / dispute / other
        return 1, 0.50


def run_prediction(trip: TripFeatures) -> tuple:
    """Try Spark, fall back to rules. Returns (pred, conf, method, ms)."""
    t0 = time.time()

    # Only attempt Spark if model files exist
    if os.path.exists(MODELS_PATH) and os.path.exists(PIPELINE_PATH):
        try:
            pred, conf = spark_predict(trip)
            method = "spark_model"
        except Exception:
            pred, conf = rule_based_predict(trip)
            method = "rule_based_fallback"
    else:
        pred, conf = rule_based_predict(trip)
        method = "rule_based_fallback"

    ms = round((time.time() - t0) * 1000, 1)
    return pred, conf, method, ms


def build_response(trip: TripFeatures, pred: int, conf: float,
                   method: str, ms: float) -> PredictionResponse:
    return PredictionResponse(
        prediction=pred,
        prediction_label="Generous Tipper" if pred == 1 else "Not a Generous Tipper",
        confidence=round(conf, 4),
        confidence_pct=f"{conf*100:.1f}%",
        inference_method=method,
        inference_time_ms=ms,
        input_summary={
            "payment_type": {1:"Credit Card",2:"Cash",3:"No Charge",
                             4:"Dispute",5:"Unknown",6:"Voided"}.get(trip.payment_type, str(trip.payment_type)),
            "trip_distance_mi": trip.trip_distance,
            "fare_amount_usd":  trip.fare_amount,
            "duration_min":     trip.trip_duration_min,
            "pickup_hour":      trip.pickup_hour,
            "time_of_day":      get_time_of_day(trip.pickup_hour),
            "is_airport_trip":  bool(trip.is_airport_trip),
            "is_weekend":       trip.pickup_dayofweek in [1, 7],
        }
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "NYC Taxi Tip Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health":       "GET  /health",
            "model_info":   "GET  /model/info",
            "stats":        "GET  /stats",
            "predict":      "POST /predict",
            "batch":        "POST /predict/batch",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    spark_available = False
    if os.path.exists(MODELS_PATH) and os.path.exists(PIPELINE_PATH):
        try:
            from pyspark.sql import SparkSession
            spark_available = True
        except ImportError:
            spark_available = False

    return HealthResponse(
        status="healthy",
        spark_model_available=spark_available,
        models_path_exists=os.path.exists(MODELS_PATH),
        pipeline_path_exists=os.path.exists(PIPELINE_PATH),
        api_version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    return ModelInfoResponse(
        model_type="RandomForestClassificationModel",
        algorithm="Random Forest",
        task="Binary Classification — generous_tipper (tip > 20% of fare)",
        training_rows=7_146_390,
        num_trees=50,
        max_depth=10,
        feature_vector_size=26,
        accuracy=0.7997,
        auc_roc=0.7888,
        f1_score=0.7784,
        precision=0.8487,
        recall=0.7996,
        top_feature="payment_type",
        top_feature_importance=0.9343,
        tuning_method="3-fold CrossValidator, 8-combo ParamGrid (24 fits)",
        model_path=MODELS_PATH
    )


@app.get("/stats", tags=["General"])
def pipeline_stats():
    return {
        "etl": {
            "raw_rows":         9_384_487,
            "clean_rows":       8_934_307,
            "rows_removed":     450_180,
            "removal_pct":      4.8,
            "runtime_seconds":  79,
            "partitions":       3,
            "features_engineered": 10,
        },
        "ml": {
            "train_rows":       7_146_390,
            "test_rows":        1_787_917,
            "train_test_split": "80/20",
            "classifier": {
                "model":        "Random Forest (tuned)",
                "num_trees":    50,
                "max_depth":    10,
                "accuracy":     0.7997,
                "auc_roc":      0.7888,
                "f1_score":     0.7784,
                "confusion_matrix": {
                    "TN": 327_423,
                    "FP": 358_154,
                    "FN": 59,
                    "TP": 1_102_281
                }
            },
            "regressor": {
                "model":        "Linear Regression",
                "target":       "fare_amount",
                "r2":           0.943,
                "rmse_usd":     4.04,
                "mae_usd":      1.38,
            }
        },
        "optimizations": {
            "column_pruning_speedup":    "6.0x",
            "partition_pruning_speedup": "2.1x",
            "aqe_speedup":              "1.4x",
        },
        "full_pipeline_runtime_seconds": 181.5,
        "dataset": "NYC TLC Yellow Taxi 2023 (Jan-Mar)"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(trip: TripFeatures):
    """
    Predict whether a taxi passenger will be a generous tipper (tip > 20% of fare).

    - **payment_type**: The most important feature (93.4% importance). Cash passengers almost never tip electronically.
    - **trip_distance / fare_amount / duration**: Secondary signals.
    - Returns prediction (0/1), confidence score, and inference method used.
    """
    try:
        pred, conf, method, ms = run_prediction(trip)
        return build_response(trip, pred, conf, method, ms)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """
    Run predictions for up to 100 trips in one request.
    Each trip is predicted independently.
    """
    if len(request.trips) == 0:
        raise HTTPException(status_code=400, detail="trips list cannot be empty")

    t0 = time.time()
    results = []
    for trip in request.trips:
        try:
            pred, conf, method, ms = run_prediction(trip)
            results.append(build_response(trip, pred, conf, method, ms))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for trip index {len(results)}: {str(e)}"
            )

    total_ms = round((time.time() - t0) * 1000, 1)
    return BatchPredictionResponse(
        total=len(results),
        predictions=results,
        inference_time_ms=total_ms
    )

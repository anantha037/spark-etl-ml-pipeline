"""
full_pipeline.py
Phase 11 - Complete Pipeline: Raw Data → ETL → ML → Predictions
The single script that runs the entire project end-to-end.
"""

import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "etl"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import PipelineModel
from pyspark import StorageLevel

from extract import run_extract
from transform import (remove_duplicates, filter_invalid_rows,
                       handle_nulls, engineer_features, select_final_columns)
from data_prep import build_feature_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_PATH   = "/home/anantha/spark-etl-ml-project/data/processed"
ML_DATA_PATH     = "/home/anantha/spark-etl-ml-project/data/ml"
MODELS_PATH      = "/home/anantha/spark-etl-ml-project/models"
OUTPUT_PATH      = "/home/anantha/spark-etl-ml-project/data/output"
POSTGRES_URL     = "jdbc:postgresql://localhost:5432/nyc_taxi"
POSTGRES_PROPS   = {
    "user": "anantha", "password": "spark123",
    "driver": "org.postgresql.Driver"
}


def get_spark() -> SparkSession:
    return SparkSession.builder \
        .appName("NYC_Taxi_Full_Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.jars",
                "/home/anantha/spark-etl-ml-project/jars/postgresql-42.7.3.jar") \
        .getOrCreate()


def banner(title: str):
    log.info("")
    log.info("╔" + "═" * 53 + "╗")
    log.info(f"║  {title:<51}║")
    log.info("╚" + "═" * 53 + "╝")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — ETL
# ══════════════════════════════════════════════════════════════════════════
def stage_etl(spark: SparkSession) -> tuple:
    banner("STAGE 1: ETL — Extract → Transform → Load")
    t = time.time()

    # Extract
    taxi_df, _ = run_extract(spark)
    taxi_df.persist(StorageLevel.MEMORY_AND_DISK)
    raw_count = taxi_df.count()
    log.info(f"[ETL] Extracted: {raw_count:,} rows")

    # Transform (lazy)
    clean_df = remove_duplicates(taxi_df)
    clean_df = filter_invalid_rows(clean_df)
    clean_df = handle_nulls(clean_df)
    clean_df = engineer_features(clean_df)
    clean_df = select_final_columns(clean_df)
    log.info("[ETL] Transform plan built")

    # Load — write to Parquet (skip if exists)
    if os.path.exists(PROCESSED_PATH) and os.listdir(PROCESSED_PATH):
        log.info("[ETL] Parquet already exists — skipping write")
        clean_df = spark.read.parquet(PROCESSED_PATH)
    else:
        clean_df.write.mode("overwrite") \
            .partitionBy("source_month") \
            .parquet(PROCESSED_PATH)
        clean_df = spark.read.parquet(PROCESSED_PATH)

    final_count = clean_df.count()
    duration = round(time.time() - t, 1)

    log.info(f"[ETL] Raw: {raw_count:,} → Clean: {final_count:,} "
             f"({raw_count-final_count:,} removed) in {duration}s")
    return clean_df, raw_count, final_count


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — ML Feature Prep
# ══════════════════════════════════════════════════════════════════════════
def stage_ml_prep(spark: SparkSession, clean_df: DataFrame) -> tuple:
    banner("STAGE 2: ML — Feature Engineering")
    t = time.time()

    # Drop leakage columns
    leakage = ["tip_amount", "tip_pct", "total_amount",
               "fare_per_mile", "source_month"]
    ml_df = clean_df.select(
        [c for c in clean_df.columns if c not in leakage]
    ).dropna()

    # Load or fit feature pipeline
    pipeline_path = f"{MODELS_PATH}/feature_pipeline"
    if os.path.exists(pipeline_path):
        log.info("[ML Prep] Loading saved feature pipeline...")
        fitted_pipeline = PipelineModel.load(pipeline_path)
        ml_df = fitted_pipeline.transform(ml_df)
    else:
        log.info("[ML Prep] Fitting new feature pipeline...")
        pipeline = build_feature_pipeline(ml_df)
        fitted_pipeline = pipeline.fit(ml_df)
        ml_df = fitted_pipeline.transform(ml_df)
        fitted_pipeline.write().overwrite().save(pipeline_path)

    ml_df = ml_df.select("features", "generous_tipper")

    # Use saved train/test or create new split
    train_path = f"{ML_DATA_PATH}/train"
    test_path  = f"{ML_DATA_PATH}/test"

    if os.path.exists(train_path) and os.path.exists(test_path):
        log.info("[ML Prep] Loading saved train/test splits...")
        train_df = spark.read.parquet(train_path)
        test_df  = spark.read.parquet(test_path)
    else:
        train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
        train_df.write.mode("overwrite").parquet(train_path)
        test_df.write.mode("overwrite").parquet(test_path)

    duration = round(time.time() - t, 1)
    log.info(f"[ML Prep] Train: {train_df.count():,} | Test: {test_df.count():,} in {duration}s")
    return train_df, test_df, ml_df


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Predict on NEW data (simulates real-time inference)
# ══════════════════════════════════════════════════════════════════════════
def stage_predict(spark: SparkSession, test_df: DataFrame) -> DataFrame:
    banner("STAGE 3: ML — Load Model & Generate Predictions")
    t = time.time()

    # Load the best tuned model
    model_path = f"{MODELS_PATH}/rf_tuned_best"
    log.info(f"[Predict] Loading model from {model_path}...")
    model = RandomForestClassificationModel.load(model_path)

    # Generate predictions
    log.info("[Predict] Running inference on test set...")
    predictions = model.transform(test_df)

    # Summary stats on predictions
    total = predictions.count()
    pred_generous = predictions.filter(F.col("prediction") == 1.0).count()
    pred_not      = total - pred_generous
    correct       = predictions.filter(
        F.col("prediction") == F.col("generous_tipper").cast("double")
    ).count()
    accuracy = round(correct / total, 4)

    duration = round(time.time() - t, 1)

    log.info(f"[Predict] Predictions generated in {duration}s")
    log.info(f"[Predict] Total predictions   : {total:,}")
    log.info(f"[Predict] Predicted generous  : {pred_generous:,} ({round(pred_generous/total*100,1)}%)")
    log.info(f"[Predict] Predicted not-generous: {pred_not:,} ({round(pred_not/total*100,1)}%)")
    log.info(f"[Predict] Accuracy             : {accuracy} ({accuracy*100:.1f}%)")

    return predictions


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — Save Predictions + Write to PostgreSQL
# ══════════════════════════════════════════════════════════════════════════
def stage_save_predictions(spark: SparkSession,
                           predictions: DataFrame,
                           clean_df: DataFrame):
    banner("STAGE 4: Save Predictions → Parquet + PostgreSQL")
    t = time.time()

    # Save predictions to Parquet
    pred_path = f"{OUTPUT_PATH}/predictions"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # probability is a Vector — extract confidence for class 1 (generous)
    # vector_to_array converts Vector → array, then [1] gets class-1 probability
    from pyspark.ml.functions import vector_to_array

    preds_with_confidence = predictions.withColumn(
        "confidence",
        F.round(vector_to_array(F.col("probability"))[1], 4)
    )

    preds_with_confidence.select(
        "generous_tipper", "prediction", "confidence"
    ).write.mode("overwrite").parquet(pred_path)
    log.info(f"[Save] Predictions saved → {pred_path}")

    # Sample — show 10 real prediction examples
    log.info("[Save] Sample predictions:")
    preds_with_confidence.select(
        "generous_tipper", "prediction", "confidence"
    ).show(10, truncate=False)

    # Write prediction summary to PostgreSQL
    total = preds_with_confidence.count()

    summary = preds_with_confidence.agg(
        F.count("*")
         .alias("total_predictions"),
        F.sum((F.col("prediction") == 1.0).cast("int"))
         .alias("predicted_generous"),
        F.sum((F.col("prediction") == 0.0).cast("int"))
         .alias("predicted_not_generous"),
        F.round(
            F.avg((F.col("prediction") ==
                   F.col("generous_tipper").cast("double")).cast("int")), 4)
         .alias("accuracy"),
        F.round(F.avg("confidence"), 4)    # ← now works: confidence is a double
         .alias("avg_confidence"),
    )

    summary_row = summary.collect()[0].asDict()
    log.info("[Save] Summary computed:")
    for k, v in summary_row.items():
        log.info(f"         {k}: {v}")

    try:
        summary.write.mode("overwrite").jdbc(
            url=POSTGRES_URL, table="prediction_summary",
            properties=POSTGRES_PROPS
        )
        log.info("[Save] Prediction summary → PostgreSQL")
    except Exception as e:
        log.warning(f"[Save] Postgres write skipped: {e}")
        fallback = f"{OUTPUT_PATH}/prediction_summary"
        os.makedirs(fallback, exist_ok=True)
        import csv
        with open(f"{fallback}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_row.keys())
            w.writeheader()
            w.writerow(summary_row)
        log.info(f"[Save] Summary saved to CSV → {fallback}/summary.csv ")

    duration = round(time.time() - t, 1)
    log.info(f"[Save] Complete in {duration}s")


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_full_pipeline():
    pipeline_start = time.time()

    banner("NYC TAXI — FULL PIPELINE  (ETL + ML + PREDICTIONS)")
    log.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    timings = {}

    # Stage 1 — ETL
    t = time.time()
    clean_df, raw_count, clean_count = stage_etl(spark)
    timings["ETL"] = round(time.time() - t, 1)

    # Stage 2 — ML Prep
    t = time.time()
    train_df, test_df, ml_df = stage_ml_prep(spark, clean_df)
    timings["ML Prep"] = round(time.time() - t, 1)

    # Stage 3 — Predict
    t = time.time()
    predictions = stage_predict(spark, test_df)
    timings["Predict"] = round(time.time() - t, 1)

    # Stage 4 — Save
    t = time.time()
    stage_save_predictions(spark, predictions, clean_df)
    timings["Save"] = round(time.time() - t, 1)

    # ── Final Report ───────────────────────────────────────────────────────
    total_time = round(time.time() - pipeline_start, 1)
    banner("FULL PIPELINE COMPLETE")
    log.info(f"  {'Stage':<12} {'Duration':>10}")
    log.info(f"  {'─'*24}")
    for stage, dur in timings.items():
        log.info(f"  {stage:<12} {dur:>9}s")
    log.info(f"  {'─'*24}")
    log.info(f"  {'TOTAL':<12} {total_time:>9}s  ({round(total_time/60,1)} min)")
    log.info("")
    log.info(f"  Raw data         : {raw_count:,} rows")
    log.info(f"  After ETL clean  : {clean_count:,} rows")
    log.info(f"  Rows removed     : {raw_count-clean_count:,} ({(raw_count-clean_count)/raw_count*100:.1f}%)")
    log.info(f"  Model            : Random Forest (tuned)")
    log.info(f"  Test predictions : {test_df.count():,} rows")
    log.info("")
    log.info("  Output locations:")
    log.info(f"    Parquet (clean)  : {PROCESSED_PATH}")
    log.info(f"    Parquet (predict): {OUTPUT_PATH}/predictions")
    log.info(f"    PostgreSQL       : trip_summary + prediction_summary")
    log.info(f"    Models saved     : {MODELS_PATH}/")
    banner("END")

    spark.stop()


if __name__ == "__main__":
    run_full_pipeline()


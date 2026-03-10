"""
data_prep.py
Phase 7 - ML Data Preparation:
Load cleaned Parquet → encode features → split train/test
"""

import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

PROCESSED_PATH = "/home/anantha/spark-etl-ml-project/data/processed"
ML_DATA_PATH   = "/home/anantha/spark-etl-ml-project/data/ml"


def get_spark(app_name="NYC_Taxi_ML_Prep") -> SparkSession:
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_clean_data(spark: SparkSession) -> DataFrame:
    """Load the processed Parquet data from Load phase."""
    df = spark.read.parquet(PROCESSED_PATH)
    log.info(f"Loaded {df.count():,} rows from processed Parquet")
    return df


def select_ml_features(df: DataFrame) -> DataFrame:
    """
    Select only the features relevant for ML.
    Drop fare breakdown columns that would leak the target
    (tip_amount directly determines generous_tipper).
    """
    # IMPORTANT: Drop tip_amount and tip_pct — they directly encode
    # the target variable. Keeping them = data leakage = fake 100% accuracy
    leakage_cols = ["tip_amount", "tip_pct", "total_amount",
                    "fare_per_mile", "source_month"]

    ml_cols = [c for c in df.columns if c not in leakage_cols]
    df = df.select(ml_cols)

    log.info(f"ML features selected: {len(ml_cols)} columns")
    log.info(f"Dropped (leakage): {leakage_cols}")
    return df


def build_feature_pipeline(df: DataFrame):
    """
    Build Spark ML Pipeline with:
    1. StringIndexer  — encode 'time_of_day' and 'store_and_fwd_flag' to numeric
    2. OneHotEncoder  — convert indexed categories to binary vectors
    3. VectorAssembler — combine all features into single 'features' vector
    4. StandardScaler — normalize numeric features (zero mean, unit variance)
    """

    # ── Categorical columns needing encoding ──────────────────────────────
    cat_cols     = ["time_of_day", "store_and_fwd_flag"]
    indexed_cols = [f"{c}_idx" for c in cat_cols]
    encoded_cols = [f"{c}_enc" for c in cat_cols]

    # Step 1: String → Index
    indexers = [
        StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
        for c, idx in zip(cat_cols, indexed_cols)
    ]

    # Step 2: Index → One-Hot Vector
    encoders = [
        OneHotEncoder(inputCol=idx, outputCol=enc)
        for idx, enc in zip(indexed_cols, encoded_cols)
    ]

    # ── Numeric columns (already numbers) ─────────────────────────────────
    numeric_cols = [
        "vendorid", "passenger_count", "trip_distance",
        "ratecodeid", "payment_type",
        "pulocationid", "dolocationid", "is_airport_trip",
        "fare_amount", "extra", "mta_tax", "tolls_amount",
        "improvement_surcharge", "congestion_surcharge", "airport_fee",
        "trip_duration_min", "pickup_hour", "pickup_dayofweek",
        "is_weekend", "avg_speed_mph",
    ]

    # Step 3: Assemble all features into one vector
    assembler = VectorAssembler(
        inputCols  = numeric_cols + encoded_cols,
        outputCol  = "features_raw",
        handleInvalid="skip"
    )

    # Step 4: Scale features (important for gradient-based models)
    scaler = StandardScaler(
        inputCol  = "features_raw",
        outputCol = "features",
        withMean  = True,
        withStd   = True
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    log.info(f"ML Pipeline built: {len(indexers)} indexers + {len(encoders)} encoders + assembler + scaler")
    return pipeline


def prepare_ml_data(spark: SparkSession):
    """
    Full ML data prep:
    1. Load clean data
    2. Remove leakage columns
    3. Drop nulls in key feature cols
    4. Fit + transform feature pipeline
    5. Split into train/test
    6. Save both splits
    """
    log.info("=" * 55)
    log.info("PHASE 7: ML DATA PREPARATION")
    log.info("=" * 55)

    # Load and prep
    df = load_clean_data(spark)
    df = select_ml_features(df)

    # Drop any remaining nulls in feature columns
    before = df.count()
    df = df.dropna()
    after  = df.count()
    log.info(f"Dropped {before - after:,} rows with nulls | Remaining: {after:,}")

    # Verify class balance
    log.info("Class distribution:")
    df.groupBy("generous_tipper").count() \
      .withColumn("pct", F.round(F.col("count") / after * 100, 1)) \
      .orderBy("generous_tipper").show()

    # Build and fit pipeline
    pipeline  = build_feature_pipeline(df)
    log.info("Fitting feature pipeline on full dataset...")
    fitted    = pipeline.fit(df)
    ml_df     = fitted.transform(df)

    # Keep only what we need: features vector + label
    ml_df = ml_df.select("features", "generous_tipper")
    log.info(f"Feature vector created.")

    # Train / Test split — 80/20, fixed seed for reproducibility
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

    # Cache both splits — they'll be used multiple times by the model
    train_df.cache()
    test_df.cache()

    train_count = train_df.count()
    test_count  = test_df.count()

    log.info(f"Train set : {train_count:,} rows ({round(train_count/after*100,1)}%)")
    log.info(f"Test  set : {test_count:,} rows ({round(test_count/after*100,1)}%)")

    # Save to disk
    os.makedirs("/home/anantha/spark-etl-ml-project/data/ml", exist_ok=True)

    train_df.write.mode("overwrite").parquet(f"{ML_DATA_PATH}/train")
    test_df.write.mode("overwrite").parquet(f"{ML_DATA_PATH}/test")

    # Save pipeline model for reuse
    model_path = "/home/anantha/spark-etl-ml-project/models/feature_pipeline"
    fitted.write().overwrite().save(model_path)

    log.info("=" * 55)
    log.info("ML DATA PREP COMPLETE !!!")
    log.info(f"  Train : {train_count:,} rows → {ML_DATA_PATH}/train")
    log.info(f"  Test  : {test_count:,} rows  → {ML_DATA_PATH}/test")
    log.info(f"  Pipeline saved → {model_path}")
    log.info("=" * 55)

    return train_df, test_df, fitted


if __name__ == "__main__":
    spark = get_spark()
    train_df, test_df, pipeline = prepare_ml_data(spark)

    print("\nSample of ML-ready training data:")
    train_df.show(3, truncate=True)

    print(f"\nFeature vector size: {len(train_df.first()['features'])}")
    spark.stop()
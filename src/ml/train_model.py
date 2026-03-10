"""
train_model.py
Phase 8 - ML Model Building:
  Model 1: Random Forest Classifier (predict generous_tipper)
  Model 2: Linear Regression (predict fare_amount)
"""

import logging
import time
import sys
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml import PipelineModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

ML_DATA_PATH  = "/home/anantha/spark-etl-ml-project/data/ml"
MODELS_PATH   = "/home/anantha/spark-etl-ml-project/models"
PROCESSED_PATH= "/home/anantha/spark-etl-ml-project/data/processed"


def get_spark(app_name="NYC_Taxi_ML_Train") -> SparkSession:
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .getOrCreate()


# ══════════════════════════════════════════════════════════════════════════
# MODEL 1 — Random Forest Classifier
# Predict: will this passenger tip > 20%? (generous_tipper = 1 or 0)
# ══════════════════════════════════════════════════════════════════════════
def train_classifier(train_df: DataFrame, test_df: DataFrame):
    log.info("─" * 55)
    log.info("MODEL 1: RANDOM FOREST CLASSIFIER")
    log.info("  Task : Predict generous_tipper (0 or 1)")
    log.info("─" * 55)

    rf = RandomForestClassifier(
        labelCol    = "generous_tipper",
        featuresCol = "features",
        numTrees    = 50,       # 50 trees — good balance of speed vs accuracy
        maxDepth    = 8,        # Limit depth to prevent overfitting
        seed        = 42
    )

    log.info("Training Random Forest (50 trees, depth 8)...")
    t_start = time.time()
    model   = rf.fit(train_df)
    t_train = round(time.time() - t_start, 1)
    log.info(f"Training complete in {t_train}s")

    # Predict on test set
    predictions = model.transform(test_df)

    # ── Evaluation ────────────────────────────────────────────────────────
    # AUC-ROC
    auc_eval = BinaryClassificationEvaluator(
        labelCol="generous_tipper", rawPredictionCol="rawPrediction"
    )
    auc = round(auc_eval.evaluate(predictions), 4)

    # Accuracy, Precision, Recall, F1
    mc_eval = MulticlassClassificationEvaluator(
        labelCol="generous_tipper", predictionCol="prediction"
    )
    accuracy  = round(mc_eval.setMetricName("accuracy").evaluate(predictions),  4)
    precision = round(mc_eval.setMetricName("weightedPrecision").evaluate(predictions), 4)
    recall    = round(mc_eval.setMetricName("weightedRecall").evaluate(predictions),    4)
    f1        = round(mc_eval.setMetricName("f1").evaluate(predictions),                4)

    log.info("─" * 55)
    log.info("CLASSIFIER RESULTS")
    log.info("─" * 55)
    log.info(f"  AUC-ROC   : {auc}")
    log.info(f"  Accuracy  : {accuracy}  ({accuracy*100:.1f}%)")
    log.info(f"  Precision : {precision}")
    log.info(f"  Recall    : {recall}")
    log.info(f"  F1 Score  : {f1}")
    log.info("─" * 55)

    # Feature importance — top 10 most influential features
    log.info("Top 10 Feature Importances:")
    importances = model.featureImportances
    feature_names = [
        "vendorid","passenger_count","trip_distance","ratecodeid",
        "payment_type","pulocationid","dolocationid","is_airport_trip",
        "fare_amount","extra","mta_tax","tolls_amount",
        "improvement_surcharge","congestion_surcharge","airport_fee",
        "trip_duration_min","pickup_hour","pickup_dayofweek",
        "is_weekend","avg_speed_mph","time_of_day_enc","store_and_fwd_enc"
    ]
    fi = sorted(
        zip(feature_names[:len(importances)], importances.toArray()),
        key=lambda x: x[1], reverse=True
    )[:10]
    for name, score in fi:
        bar = "█" * int(score * 200)
        log.info(f"  {name:<25} {score:.4f}  {bar}")

    # Save model
    save_path = f"{MODELS_PATH}/random_forest_classifier"
    model.write().overwrite().save(save_path)
    log.info(f"Model saved → {save_path}")

    return model, predictions, {
        "auc": auc, "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1
    }


# ══════════════════════════════════════════════════════════════════════════
# MODEL 2 — Linear Regression
# Predict: what will the fare_amount be?
# Uses the cleaned processed data (not the leakage-safe ML data)
# ══════════════════════════════════════════════════════════════════════════
def train_regressor(spark: SparkSession):
    log.info("─" * 55)
    log.info("MODEL 2: LINEAR REGRESSION")
    log.info("  Task : Predict fare_amount")
    log.info("─" * 55)

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    # Reload processed data — use features that don't leak fare
    df = spark.read.parquet(PROCESSED_PATH)
    df = df.select(
        "trip_distance", "trip_duration_min", "pickup_hour",
        "pickup_dayofweek", "is_weekend", "is_airport_trip",
        "passenger_count", "pulocationid", "dolocationid",
        "avg_speed_mph", "fare_amount"
    ).dropna()

    assembler = VectorAssembler(
        inputCols=[
            "trip_distance", "trip_duration_min", "pickup_hour",
            "pickup_dayofweek", "is_weekend", "is_airport_trip",
            "passenger_count", "pulocationid", "dolocationid", "avg_speed_mph"
        ],
        outputCol="features", handleInvalid="skip"
    )
    df = assembler.transform(df).select("features", "fare_amount")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()

    lr = LinearRegression(
        labelCol   = "fare_amount",
        featuresCol= "features",
        maxIter    = 10,
        regParam   = 0.1,   # L2 regularization
        elasticNetParam = 0.0
    )

    log.info("Training Linear Regression...")
    t_start = time.time()
    model   = lr.fit(train_df)
    t_train = round(time.time() - t_start, 1)
    log.info(f"Training complete in {t_train}s")

    predictions = model.transform(test_df)

    # Evaluation
    reg_eval = RegressionEvaluator(
        labelCol="fare_amount", predictionCol="prediction"
    )
    rmse = round(reg_eval.setMetricName("rmse").evaluate(predictions), 4)
    mae  = round(reg_eval.setMetricName("mae").evaluate(predictions),  4)
    r2   = round(reg_eval.setMetricName("r2").evaluate(predictions),   4)

    log.info("─" * 55)
    log.info("REGRESSION RESULTS")
    log.info("─" * 55)
    log.info(f"  RMSE      : ${rmse}  (avg prediction error)")
    log.info(f"  MAE       : ${mae}   (mean absolute error)")
    log.info(f"  R²        : {r2}   (1.0 = perfect fit)")
    log.info("─" * 55)

    save_path = f"{MODELS_PATH}/linear_regression_fare"
    model.write().overwrite().save(save_path)
    log.info(f"Model saved → {save_path}")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    log.info("╔" + "═" * 53 + "╗")
    log.info("║  PHASE 8: ML MODEL BUILDING                        ║")
    log.info("╚" + "═" * 53 + "╝")

    # Load pre-prepared train/test splits from Phase 7
    log.info("Loading train/test splits from Phase 7...")
    train_df = spark.read.parquet(f"{ML_DATA_PATH}/train")
    test_df  = spark.read.parquet(f"{ML_DATA_PATH}/test")
    train_df.cache()
    test_df.cache()
    log.info(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

    # Train both models
    clf_model, clf_preds, clf_metrics = train_classifier(train_df, test_df)
    reg_model, reg_metrics            = train_regressor(spark)

    # Final summary
    log.info("╔" + "═" * 53 + "╗")
    log.info("║  PHASE 8 COMPLETE — MODEL SUMMARY                  ║")
    log.info("╚" + "═" * 53 + "╝")
    log.info("  CLASSIFIER (Random Forest — generous_tipper)")
    log.info(f"    AUC={clf_metrics['auc']}  Acc={clf_metrics['accuracy']}  F1={clf_metrics['f1']}")
    log.info("  REGRESSOR (Linear Regression — fare_amount)")
    log.info(f"    RMSE=${reg_metrics['rmse']}  MAE=${reg_metrics['mae']}  R²={reg_metrics['r2']}")

    spark.stop()
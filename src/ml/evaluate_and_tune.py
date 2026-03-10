"""
evaluate_and_tune.py
Phase 9 + 10 — Model Evaluation & Hyperparameter Tuning
"""

import logging
import time
import sys
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

ML_DATA_PATH = "/home/anantha/spark-etl-ml-project/data/ml"
MODELS_PATH  = "/home/anantha/spark-etl-ml-project/models"


def get_spark() -> SparkSession:
    return SparkSession.builder \
        .appName("NYC_Taxi_Evaluate_Tune") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .getOrCreate()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 9 — DETAILED EVALUATION
# ══════════════════════════════════════════════════════════════════════════
def evaluate_classifier(predictions: DataFrame, label: str = "generous_tipper"):
    """Full evaluation suite: confusion matrix + all metrics."""

    log.info("═" * 55)
    log.info("PHASE 9: DETAILED MODEL EVALUATION")
    log.info("═" * 55)

    total = predictions.count()

    # ── Confusion Matrix ───────────────────────────────────────────────────
    log.info("CONFUSION MATRIX:")
    log.info("  Predicted →    0 (Not generous)    1 (Generous)")
    log.info("  Actual ↓")

    cm = predictions.groupBy(label, "prediction") \
                    .count() \
                    .orderBy(label, "prediction") \
                    .collect()

    cm_dict = {(int(r[label]), int(r["prediction"])): r["count"] for r in cm}
    tn = cm_dict.get((0, 0), 0)
    fp = cm_dict.get((0, 1), 0)
    fn = cm_dict.get((1, 0), 0)
    tp = cm_dict.get((1, 1), 0)

    log.info(f"  Actual 0   |  TN={tn:>9,}    FP={fp:>9,}")
    log.info(f"  Actual 1   |  FN={fn:>9,}    TP={tp:>9,}")

    # ── Manual metric calculation ──────────────────────────────────────────
    precision_0 = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0
    precision_1 = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
    recall_0    = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    recall_1    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    f1_0 = round(2 * precision_0 * recall_0 / (precision_0 + recall_0), 4) if (precision_0 + recall_0) > 0 else 0
    f1_1 = round(2 * precision_1 * recall_1 / (precision_1 + recall_1), 4) if (precision_1 + recall_1) > 0 else 0
    accuracy = round((tp + tn) / total, 4)

    log.info("─" * 55)
    log.info("PER-CLASS METRICS:")
    log.info(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    log.info(f"  {'─'*52}")
    log.info(f"  {'0 (Not tip)':<12} {precision_0:>10} {recall_0:>10} {f1_0:>10} {tn+fp:>10,}")
    log.info(f"  {'1 (Tipper)':<12} {precision_1:>10} {recall_1:>10} {f1_1:>10} {tp+fn:>10,}")
    log.info("─" * 55)
    log.info(f"  Overall Accuracy : {accuracy} ({accuracy*100:.1f}%)")
    log.info(f"  Total Samples    : {total:,}")

    # ── AUC ───────────────────────────────────────────────────────────────
    auc_eval = BinaryClassificationEvaluator(labelCol=label)
    auc = round(auc_eval.evaluate(predictions), 4)
    log.info(f"  AUC-ROC          : {auc}")
    log.info("─" * 55)

    # ── Prediction distribution ────────────────────────────────────────────
    log.info("PREDICTION DISTRIBUTION:")
    predictions.groupBy("prediction").count() \
        .withColumn("pct", F.round(F.col("count") / total * 100, 1)) \
        .orderBy("prediction").show()

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "auc": auc,
        "precision_1": precision_1, "recall_1": recall_1, "f1_1": f1_1
    }


# ══════════════════════════════════════════════════════════════════════════
# PHASE 10 — HYPERPARAMETER TUNING (CrossValidator)
# ══════════════════════════════════════════════════════════════════════════
def tune_classifier(train_df: DataFrame, test_df: DataFrame):
    """
    Grid search over Random Forest hyperparameters using 3-fold CV.
    We use a sample for tuning to keep runtime manageable.
    """
    log.info("═" * 55)
    log.info("PHASE 10: HYPERPARAMETER TUNING")
    log.info("  Method : 3-Fold Cross Validation + Grid Search")
    log.info("═" * 55)

    # Use 20% sample for tuning — full 7M rows × CV is too slow on laptop
    tune_df = train_df.sample(fraction=0.20, seed=42)
    tune_count = tune_df.count()
    log.info(f"Tuning sample: {tune_count:,} rows (20% of train set)")

    rf = RandomForestClassifier(
        labelCol="generous_tipper",
        featuresCol="features",
        seed=42
    )

    # Parameter grid — 2×2×2 = 8 combinations × 3 folds = 24 model fits
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees,  [30, 50]) \
        .addGrid(rf.maxDepth,  [6, 10]) \
        .addGrid(rf.minInstancesPerNode, [1, 5]) \
        .build()

    evaluator = BinaryClassificationEvaluator(
        labelCol="generous_tipper",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator          = rf,
        estimatorParamMaps = param_grid,
        evaluator          = evaluator,
        numFolds           = 3,
        seed               = 42,
        parallelism        = 2   # run 2 models in parallel
    )

    log.info(f"Grid: {len(param_grid)} combos × 3 folds = {len(param_grid)*3} fits")
    log.info("Running cross-validation (this takes ~5-8 min)...")

    t_start  = time.time()
    cv_model = cv.fit(tune_df)
    t_cv     = round(time.time() - t_start, 1)
    log.info(f"Cross-validation complete in {t_cv}s")

    # Best model parameters
    best_model  = cv_model.bestModel
    best_trees  = best_model.getNumTrees
    best_depth  = best_model.getOrDefault("maxDepth")
    best_min    = best_model.getOrDefault("minInstancesPerNode")
    cv_scores   = cv_model.avgMetrics

    log.info("─" * 55)
    log.info("CROSS-VALIDATION RESULTS (AUC per combo):")
    param_labels = [
        f"trees={p[rf.numTrees]}, depth={p[rf.maxDepth]}, minInst={p[rf.minInstancesPerNode]}"
        for p in param_grid
    ]
    for label, score in zip(param_labels, cv_scores):
        bar = "█" * int(score * 50)
        log.info(f"  {label:<40} AUC={score:.4f}  {bar}")

    log.info("─" * 55)
    log.info(f"  Best params : numTrees={best_trees}, maxDepth={best_depth}, minInstances={best_min}")
    log.info(f"  Best AUC    : {round(max(cv_scores), 4)}")

    # ── Evaluate best model on full test set ──────────────────────────────
    log.info("Evaluating best model on full test set...")
    best_preds  = best_model.transform(test_df)

    mc_eval  = MulticlassClassificationEvaluator(
        labelCol="generous_tipper", predictionCol="prediction"
    )
    auc_eval = BinaryClassificationEvaluator(labelCol="generous_tipper")

    final_acc = round(mc_eval.setMetricName("accuracy").evaluate(best_preds),  4)
    final_f1  = round(mc_eval.setMetricName("f1").evaluate(best_preds),        4)
    final_auc = round(auc_eval.evaluate(best_preds),                           4)

    log.info("─" * 55)
    log.info("TUNED MODEL vs BASELINE:")
    log.info(f"  {'Metric':<12} {'Baseline':>12} {'Tuned':>12} {'Δ':>8}")
    log.info(f"  {'─'*46}")
    log.info(f"  {'Accuracy':<12} {'0.7996':>12} {final_acc:>12} {final_acc-0.7996:>+8.4f}")
    log.info(f"  {'F1 Score':<12} {'0.7782':>12} {final_f1:>12} {final_f1-0.7782:>+8.4f}")
    log.info(f"  {'AUC-ROC':<12} {'0.7861':>12} {final_auc:>12} {final_auc-0.7861:>+8.4f}")
    log.info("─" * 55)

    # Save best model
    save_path = f"{MODELS_PATH}/rf_tuned_best"
    best_model.write().overwrite().save(save_path)
    log.info(f"Tuned model saved → {save_path}")

    return best_model, {
        "accuracy": final_acc, "f1": final_f1, "auc": final_auc,
        "best_trees": best_trees, "best_depth": best_depth
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    # Load train/test
    log.info("Loading train/test data...")
    train_df = spark.read.parquet(f"{ML_DATA_PATH}/train")
    test_df  = spark.read.parquet(f"{ML_DATA_PATH}/test")
    train_df.cache()
    test_df.cache()

    # Load baseline model from Phase 8
    log.info("Loading baseline Random Forest model...")
    baseline_model = RandomForestClassificationModel.load(
        f"{MODELS_PATH}/random_forest_classifier"
    )

    # Phase 9 — Detailed evaluation of baseline
    baseline_preds = baseline_model.transform(test_df)
    baseline_eval  = evaluate_classifier(baseline_preds)

    # Phase 10 — Tune and compare
    best_model, tuned_eval = tune_classifier(train_df, test_df)

    # Final summary
    log.info("╔" + "═" * 53 + "╗")
    log.info("║  PHASES 9+10 COMPLETE                              ║")
    log.info("╚" + "═" * 53 + "╝")
    log.info(f"  Baseline → Acc:{baseline_eval['accuracy']} AUC:{baseline_eval['auc']} F1:{baseline_eval['f1_1']}")
    log.info(f"  Tuned    → Acc:{tuned_eval['accuracy']}  AUC:{tuned_eval['auc']}  F1:{tuned_eval['f1']}")
    log.info(f"  Best config: numTrees={tuned_eval['best_trees']}, maxDepth={tuned_eval['best_depth']}")

    spark.stop()
"""
etl_optimizer.py
Phase 6 - ETL Optimization: Measures and compares pipeline performance
using key Spark optimization techniques.
"""

import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "etl"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark import StorageLevel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

PROCESSED_PATH = "/home/anantha/spark-etl-ml-project/data/processed"


def get_spark(app_name="NYC_Taxi_Optimizer") -> SparkSession:
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


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 1 — Partition Pruning
# Spark only reads the partitions it needs instead of all data
# ══════════════════════════════════════════════════════════════════════════
def test_partition_pruning(spark: SparkSession):
    log.info("─" * 55)
    log.info("OPT 1: PARTITION PRUNING")
    log.info("─" * 55)

    df = spark.read.parquet(PROCESSED_PATH)

    # WITHOUT pruning — scans all 3 partitions
    t1 = time.time()
    count_all = df.filter(F.col("source_month") == "2023-01").count()
    t_without = round(time.time() - t1, 2)

    # WITH pruning — Spark knows to only read source_month=2023-01/ folder
    # This works automatically because we partitioned by source_month on write
    t2 = time.time()
    df_pruned = spark.read.parquet(f"{PROCESSED_PATH}/source_month=2023-01")
    count_pruned = df_pruned.count()
    t_with = round(time.time() - t2, 2)

    log.info(f"  Without pruning : {t_without}s → {count_all:,} rows (reads all partitions)")
    log.info(f"  With pruning    : {t_with}s → {count_pruned:,} rows (reads 1 partition only)")
    log.info(f"  Speedup         : {round(t_without/t_with, 1)}x faster")

    return {"without": t_without, "with": t_with}


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 2 — Caching / Persistence
# Avoid re-computing the same DataFrame multiple times
# ══════════════════════════════════════════════════════════════════════════
def test_caching(spark: SparkSession):
    log.info("─" * 55)
    log.info("OPT 2: CACHING vs NO CACHE")
    log.info("─" * 55)

    df = spark.read.parquet(PROCESSED_PATH)

    # WITHOUT cache — each action re-reads from disk
    t1 = time.time()
    _ = df.filter(F.col("generous_tipper") == 1).count()
    _ = df.filter(F.col("is_airport_trip") == 1).count()
    _ = df.agg(F.avg("fare_amount")).collect()
    t_without = round(time.time() - t1, 2)

    # WITH cache — first action loads into memory, rest served from RAM
    df_cached = spark.read.parquet(PROCESSED_PATH)
    df_cached.persist(StorageLevel.MEMORY_AND_DISK)
    df_cached.count()  # warm the cache

    t2 = time.time()
    _ = df_cached.filter(F.col("generous_tipper") == 1).count()
    _ = df_cached.filter(F.col("is_airport_trip") == 1).count()
    _ = df_cached.agg(F.avg("fare_amount")).collect()
    t_with = round(time.time() - t2, 2)

    df_cached.unpersist()

    log.info(f"  Without cache   : {t_without}s (3 actions × full disk read)")
    log.info(f"  With cache      : {t_with}s (3 actions × RAM read)")
    log.info(f"  Speedup         : {round(t_without/t_with, 1)}x faster")

    return {"without": t_without, "with": t_with}


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 3 — Column Pruning (Select only needed columns)
# Less data shuffled across network and memory
# ══════════════════════════════════════════════════════════════════════════
def test_column_pruning(spark: SparkSession):
    log.info("─" * 55)
    log.info("OPT 3: COLUMN PRUNING")
    log.info("─" * 55)

    df = spark.read.parquet(PROCESSED_PATH)

    # WITHOUT column pruning — carries all 28 columns
    t1 = time.time()
    result1 = df.groupBy("pickup_hour") \
                .agg(F.avg("tip_pct").alias("avg_tip")) \
                .orderBy("pickup_hour") \
                .collect()
    t_without = round(time.time() - t1, 2)

    # WITH column pruning — only load columns we actually need
    t2 = time.time()
    result2 = df.select("pickup_hour", "tip_pct") \
                .groupBy("pickup_hour") \
                .agg(F.avg("tip_pct").alias("avg_tip")) \
                .orderBy("pickup_hour") \
                .collect()
    t_with = round(time.time() - t2, 2)

    log.info(f"  Without pruning : {t_without}s (28 cols in memory)")
    log.info(f"  With pruning    : {t_with}s (2 cols in memory)")
    log.info(f"  Speedup         : {round(t_without/t_with, 1)}x faster")

    return {"without": t_without, "with": t_with}


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 4 — Adaptive Query Execution (AQE)
# Spark re-optimizes the query plan at runtime based on actual data stats
# ══════════════════════════════════════════════════════════════════════════
def test_aqe(spark: SparkSession):
    log.info("─" * 55)
    log.info("OPT 4: ADAPTIVE QUERY EXECUTION (AQE)")
    log.info("─" * 55)

    df = spark.read.parquet(PROCESSED_PATH)

    # WITHOUT AQE
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    t1 = time.time()
    df.groupBy("time_of_day", "payment_type") \
      .agg(F.count("*"), F.avg("fare_amount")) \
      .collect()
    t_without = round(time.time() - t1, 2)

    # WITH AQE — Spark dynamically coalesces shuffle partitions,
    # switches join strategies, and handles skewed data at runtime
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    t2 = time.time()
    df.groupBy("time_of_day", "payment_type") \
      .agg(F.count("*"), F.avg("fare_amount")) \
      .collect()
    t_with = round(time.time() - t2, 2)

    log.info(f"  Without AQE     : {t_without}s")
    log.info(f"  With AQE        : {t_with}s")
    log.info(f"  Speedup         : {round(t_without/t_with, 1)}x faster")

    return {"without": t_without, "with": t_with}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    spark = get_spark()

    log.info("╔" + "═" * 53 + "╗")
    log.info("║  PHASE 6: ETL OPTIMIZATION BENCHMARKS              ║")
    log.info("╚" + "═" * 53 + "╝")

    r1 = test_partition_pruning(spark)
    r2 = test_caching(spark)
    r3 = test_column_pruning(spark)
    r4 = test_aqe(spark)

    log.info("═" * 55)
    log.info("OPTIMIZATION SUMMARY")
    log.info("═" * 55)
    log.info(f"  Partition Pruning : {r1['without']}s → {r1['with']}s  ({round(r1['without']/max(r1['with'],0.1),1)}x)")
    log.info(f"  Caching           : {r2['without']}s → {r2['with']}s  ({round(r2['without']/max(r2['with'],0.1),1)}x)")
    log.info(f"  Column Pruning    : {r3['without']}s → {r3['with']}s  ({round(r3['without']/max(r3['with'],0.1),1)}x)")
    log.info(f"  AQE               : {r4['without']}s → {r4['with']}s  ({round(r4['without']/max(r4['with'],0.1),1)}x)")
    log.info("═" * 55)
    log.info("These optimizations are already applied in our pipeline")

    spark.stop()
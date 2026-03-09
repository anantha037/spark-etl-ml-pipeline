"""
load.py
Phase 4 - Load: Write transformed data to:
  1. Parquet (partitioned by month) — acts as our data warehouse
  2. PostgreSQL — relational database sink
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from transform import run_transform

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths & Config ─────────────────────────────────────────────────────────
PROCESSED_PATH = "/home/anantha/spark-etl-ml-project/data/processed"
POSTGRES_URL   = "jdbc:postgresql://localhost:5432/nyc_taxi"
POSTGRES_PROPS = {
    "user"    : "anantha",
    "password": "spark123",
    "driver"  : "org.postgresql.Driver"
}


# ══════════════════════════════════════════════════════════════════════════
# SINK 1 — Write to Parquet (partitioned by source_month)
# ══════════════════════════════════════════════════════════════════════════
def write_parquet(df: DataFrame, path: str) -> int:
    """
    Write DataFrame to partitioned Parquet files.
    Partitioning by source_month lets Spark skip irrelevant partitions
    when future queries filter by month — this is partition pruning.
    """
    import os
    # Skip writing if already exists — saves 5 minutes on re-runs
    if os.path.exists(path) and os.listdir(path):
        log.info(f"[Parquet] Already exists at {path} — skipping write ✅")
        written_df = df.sparkSession.read.parquet(path)
        row_count = written_df.count()
        log.info(f"[Parquet] Verified: {row_count:,} rows on disk")
        written_df.groupBy("source_month").count().orderBy("source_month").show()
        return row_count
    
    log.info(f"[Parquet] Writing to: {path}")
    log.info(f"[Parquet] Partitioning by: source_month")

    df.write \
      .mode("overwrite") \
      .partitionBy("source_month") \
      .parquet(path)

    # Verify what was written
    written_df = df.sparkSession.read.parquet(path)
    row_count  = written_df.count()

    log.info(f"[Parquet] ✅ Written: {row_count:,} rows")
    log.info(f"[Parquet] Partitions created:")

    written_df.groupBy("source_month") \
              .count() \
              .orderBy("source_month") \
              .show()

    return row_count


# ══════════════════════════════════════════════════════════════════════════
# SINK 2 — Write to PostgreSQL
# ══════════════════════════════════════════════════════════════════════════
def write_postgres(df: DataFrame, table: str, url: str, props: dict) -> None:
    """
    Write a summary aggregation to PostgreSQL.
    We write a monthly summary table rather than all 9M rows
    because loading 9M rows via JDBC is extremely slow.
    The Parquet file IS our data warehouse for full data.
    PostgreSQL holds the aggregated reporting table.
    """
    log.info(f"[Postgres] Building summary table: {table}")

    summary_df = df.groupBy("source_month", "time_of_day", "is_airport_trip") \
        .agg(
            F.count("*")                          .alias("total_trips"),
            F.round(F.avg("fare_amount"),    2)   .alias("avg_fare"),
            F.round(F.avg("tip_amount"),     2)   .alias("avg_tip"),
            F.round(F.avg("trip_distance"),  2)   .alias("avg_distance"),
            F.round(F.avg("trip_duration_min"), 2).alias("avg_duration_min"),
            F.round(F.avg("tip_pct"),        4)   .alias("avg_tip_pct"),
            F.round(F.avg("generous_tipper"),3)   .alias("generous_tipper_rate"),
            F.sum("generous_tipper")              .alias("generous_tippers"),
        ) \
        .orderBy("source_month", "time_of_day")

    log.info(f"[Postgres] Summary rows to write: {summary_df.count()}")

    try:
        summary_df.write \
            .mode("overwrite") \
            .jdbc(url=url, table=table, properties=props)
        log.info(f"[Postgres] ✅ Written to table: {table}")
        summary_df.show(20, truncate=False)

    except Exception as e:
        log.warning(f"[Postgres] ⚠️  Could not write to PostgreSQL: {e}")
        log.warning("[Postgres] Saving summary as CSV fallback instead...")

        fallback_path = "/home/anantha/spark-etl-ml-project/data/output/summary"
        summary_df.coalesce(1) \
                  .write \
                  .mode("overwrite") \
                  .option("header", "true") \
                  .csv(fallback_path)
        log.info(f"[Postgres] ✅ Summary saved to CSV: {fallback_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════
def run_load(spark: SparkSession) -> DataFrame:

    log.info("=" * 55)
    log.info("PHASE 4: LOAD STARTED")
    log.info("=" * 55)

    # Get transformed data
    df, raw_count = run_transform(spark)

    # ── Sink 1: Parquet ────────────────────────────────────────────────────
    final_count = write_parquet(df, PROCESSED_PATH)

    # ── Reload from Parquet for Postgres sink ─────────────────────────────
    # Re-read from the written Parquet — this is now our clean source of truth
    clean_df = spark.read.parquet(PROCESSED_PATH)

    # ── Sink 2: PostgreSQL (summary table) ────────────────────────────────
    write_postgres(clean_df, "trip_summary", POSTGRES_URL, POSTGRES_PROPS)

    log.info("=" * 55)
    log.info("LOAD COMPLETE ✅")
    log.info(f"  Raw rows in     : {raw_count:,}")
    log.info(f"  Clean rows out  : {final_count:,}")
    log.info(f"  Rows removed    : {raw_count - final_count:,} ({((raw_count-final_count)/raw_count*100):.1f}%)")
    log.info(f"  Parquet path    : {PROCESSED_PATH}")
    log.info(f"  Postgres table  : trip_summary")
    log.info("=" * 55)

    return clean_df


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Load") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.jars", "/home/anantha/spark-etl-ml-project/jars/postgresql-42.7.3.jar") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    clean_df = run_load(spark)

    print("\n✅ FINAL VERIFICATION — Partition sizes:")
    clean_df.groupBy("source_month") \
            .count() \
            .orderBy("source_month") \
            .show()

    print("\n✅ SCHEMA OF LOADED DATA:")
    clean_df.printSchema()

    spark.stop()
    log.info("Done.")
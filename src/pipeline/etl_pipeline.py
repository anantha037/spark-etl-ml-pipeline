"""
etl_pipeline.py
Phase 5 - Full ETL Pipeline: Orchestrates Extract → Transform → Load
in a single executable with timing, logging, and error handling.
"""

import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "etl"))

from pyspark.sql import SparkSession
from extract   import run_extract
from transform import run_transform
from load      import run_load, write_parquet, write_postgres

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
PROCESSED_PATH = "/home/anantha/spark-etl-ml-project/data/processed"
POSTGRES_URL   = "jdbc:postgresql://localhost:5432/nyc_taxi"
POSTGRES_PROPS = {
    "user"    : "anantha",
    "password": "spark123",
    "driver"  : "org.postgresql.Driver"
}


def create_pipeline_spark_session() -> SparkSession:
    """Single Spark session shared across all pipeline phases."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_ETL_Pipeline") \
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
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def print_banner(title: str):
    """Print a section banner to make logs easy to read."""
    log.info("")
    log.info("╔" + "═" * 53 + "╗")
    log.info(f"║  {title:<51}║")
    log.info("╚" + "═" * 53 + "╝")


def run_pipeline():
    """
    Master pipeline function.
    Runs Extract → Transform → Load with:
    - Per-phase timing
    - Total pipeline timing
    - Error handling per phase (one phase failure doesn't kill everything)
    - Final summary report
    """
    pipeline_start = time.time()
    results = {
        "extract"  : {"status": "NOT RUN", "duration": 0, "rows": 0},
        "transform": {"status": "NOT RUN", "duration": 0, "rows": 0},
        "load"     : {"status": "NOT RUN", "duration": 0, "rows": 0},
    }

    print_banner("NYC TAXI ETL PIPELINE STARTING")
    log.info(f"Pipeline started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    spark = create_pipeline_spark_session()
    log.info(f"Spark {spark.version} session ready.")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — EXTRACT
    # ══════════════════════════════════════════════════════════════════════
    print_banner("PHASE 1: EXTRACT")
    phase_start = time.time()

    try:
        taxi_df, zone_df = run_extract(spark)
        raw_count = taxi_df.count()

        results["extract"]["status"]   = "SUCCESS"
        results["extract"]["rows"]     = raw_count
        results["extract"]["duration"] = round(time.time() - phase_start, 1)

        log.info(f"Extract complete — {raw_count:,} rows in {results['extract']['duration']}s")

    except Exception as e:
        results["extract"]["status"] = f"FAILED: {e}"
        log.error(f"Extract phase failed: {e}")
        spark.stop()
        _print_summary(results, time.time() - pipeline_start)
        sys.exit(1)   # Can't continue without data

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — TRANSFORM
    # ══════════════════════════════════════════════════════════════════════
    print_banner("PHASE 2: TRANSFORM")
    phase_start = time.time()

    try:
        # Pass already-loaded taxi_df directly to avoid re-reading files
        from pyspark import StorageLevel
        taxi_df.persist(StorageLevel.MEMORY_AND_DISK)

        from transform import (remove_duplicates, filter_invalid_rows,
                                handle_nulls, engineer_features,
                                select_final_columns)

        clean_df = remove_duplicates(taxi_df)
        clean_df = filter_invalid_rows(clean_df)
        clean_df = handle_nulls(clean_df)
        clean_df = engineer_features(clean_df)
        clean_df = select_final_columns(clean_df)

        results["transform"]["status"]   = "SUCCESS"
        results["transform"]["duration"] = round(time.time() - phase_start, 1)
        log.info(f"Transform complete — plan built in {results['transform']['duration']}s")

    except Exception as e:
        results["transform"]["status"] = f"FAILED: {e}"
        log.error(f"Transform phase failed: {e}")
        # Don't exit — we can still report partial results
        spark.stop()
        _print_summary(results, time.time() - pipeline_start)
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3 — LOAD
    # ══════════════════════════════════════════════════════════════════════
    print_banner("PHASE 3: LOAD")
    phase_start = time.time()

    try:
        # Write to Parquet — this is where Spark executes the full plan
        final_count = write_parquet(clean_df, PROCESSED_PATH)

        # Reload clean data and write summary to Postgres
        loaded_df = spark.read.parquet(PROCESSED_PATH)
        write_postgres(loaded_df, "trip_summary", POSTGRES_URL, POSTGRES_PROPS)

        results["load"]["status"]   = "SUCCESS"
        results["load"]["rows"]     = final_count
        results["load"]["duration"] = round(time.time() - phase_start, 1)
        log.info(f"Load complete — {final_count:,} rows written in {results['load']['duration']}s")

    except Exception as e:
        results["load"]["status"] = f"FAILED: {e}"
        log.error(f"Load phase failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_time = round(time.time() - pipeline_start, 1)
    _print_summary(results, total_time)

    spark.stop()
    log.info("Spark session stopped. Pipeline finished.")


def _print_summary(results: dict, total_time: float):
    """Print a clean pipeline execution summary."""
    raw   = results["extract"]["rows"]
    clean = results["load"]["rows"]
    removed = raw - clean if raw and clean else 0
    pct     = (removed / raw * 100) if raw else 0

    print_banner("PIPELINE EXECUTION SUMMARY")
    log.info(f"  Extract   : {results['extract']['status']:<30} | {results['extract']['duration']}s | {results['extract']['rows']:,} rows")
    log.info(f"  Transform : {results['transform']['status']:<30} | {results['transform']['duration']}s")
    log.info(f"  Load      : {results['load']['status']:<30} | {results['load']['duration']}s | {results['load']['rows']:,} rows")
    log.info("  " + "─" * 51)
    log.info(f"  Raw rows       : {raw:,}")
    log.info(f"  Clean rows     : {clean:,}")
    log.info(f"  Rows removed   : {removed:,} ({pct:.1f}%)")
    log.info(f"  Total time     : {total_time}s ({round(total_time/60, 1)} min)")
    log.info("  " + "─" * 51)
    all_ok = all("SUCCESS" in v["status"] for v in results.values())
    log.info(f"  Pipeline result: {'ALL PHASES PASSED' if all_ok else 'SOME PHASES FAILED'}")
    print_banner("END OF PIPELINE")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
"""
transform.py
Phase 3 - Transform: Clean, validate, and engineer features
from the extracted NYC Taxi DataFrame.

Fix: Removed redundant .count() calls that caused JVM OOM.
Uses a single checkpoint count instead of before/after per step.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract import create_spark_session, run_extract

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Thresholds ─────────────────────────────────────────────────────────────
MIN_FARE         = 2.50
MAX_FARE         = 500.0
MIN_DISTANCE     = 0.01
MAX_DISTANCE     = 100.0
MIN_DURATION_MIN = 0.5        # 30 seconds
MAX_DURATION_MIN = 180.0      # 3 hours
MAX_SPEED_MPH    = 100.0
VALID_PAYMENT    = [1, 2, 3, 4, 5, 6]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Remove Duplicates
# ══════════════════════════════════════════════════════════════════════════
def remove_duplicates(df: DataFrame) -> DataFrame:
    """Drop exact duplicate rows."""
    df = df.dropDuplicates()
    log.info("[Step 1] Duplicates removed ✅")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Filter Invalid Rows
# ══════════════════════════════════════════════════════════════════════════
def filter_invalid_rows(df: DataFrame) -> DataFrame:
    """
    Remove rows that violate business rules or physical reality.
    NO .count() calls here — just build the filter expression.
    Spark will execute this lazily when we finally write/count at the end.
    """
    df = df.filter(
        # Valid fare range
        (F.col("fare_amount")   >= MIN_FARE)  &
        (F.col("fare_amount")   <= MAX_FARE)  &

        # Valid distance
        (F.col("trip_distance") >= MIN_DISTANCE) &
        (F.col("trip_distance") <= MAX_DISTANCE) &

        # No negative tips or totals
        (F.col("tip_amount")    >= 0)  &
        (F.col("total_amount")  >  0)  &

        # Dropoff must be after pickup
        (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime")) &

        # Valid payment type codes
        (F.col("payment_type").isin(VALID_PAYMENT)) &

        # Only 2023 data (removes data entry errors with wrong years)
        (F.year("tpep_pickup_datetime") == 2023)
    )

    log.info("[Step 2] Invalid row filters applied ✅")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Handle Nulls
# ══════════════════════════════════════════════════════════════════════════
def handle_nulls(df: DataFrame) -> DataFrame:
    """
    Fill nulls with sensible defaults.
    - passenger_count → 1  (most common: solo rider)
    - ratecodeid      → 1  (standard rate)
    - congestion_surcharge → 2.5 (standard NYC surcharge)
    - airport_fee     → 0.0
    - store_and_fwd_flag → 'N'
    """
    df = df.fillna({
        "passenger_count"     : 1,
        "ratecodeid"          : 1,
        "congestion_surcharge": 2.5,
        "airport_fee"         : 0.0,
        "store_and_fwd_flag"  : "N",
    })

    log.info("[Step 3] Nulls filled with defaults ✅")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════
def engineer_features(df: DataFrame) -> DataFrame:
    """
    Create new columns derived from existing data.
    All operations are lazy transformations — no Spark actions triggered.
    """

    df = df \
        .withColumn(
            # How long was the trip in minutes?
            "trip_duration_min",
            F.round(
                (F.unix_timestamp("tpep_dropoff_datetime") -
                 F.unix_timestamp("tpep_pickup_datetime")) / 60.0, 2
            )
        ).withColumn(
            # What hour did the trip start? (rush hour analysis)
            "pickup_hour",
            F.hour("tpep_pickup_datetime")
        ).withColumn(
            # Day of week: 1=Sunday, 7=Saturday
            "pickup_dayofweek",
            F.dayofweek("tpep_pickup_datetime")
        ).withColumn(
            # Binary: is it a weekend?
            "is_weekend",
            F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1).otherwise(0)
        ).withColumn(
            # Bucket time into 4 periods for ML categorical feature
            "time_of_day",
            F.when(F.col("pickup_hour").between(0,  5),  "night")
             .when(F.col("pickup_hour").between(6,  11), "morning")
             .when(F.col("pickup_hour").between(12, 17), "afternoon")
             .otherwise("evening")
        ).withColumn(
            # Average speed — helps catch impossible trips
            "avg_speed_mph",
            F.round(
                F.col("trip_distance") /
                ((F.unix_timestamp("tpep_dropoff_datetime") -
                  F.unix_timestamp("tpep_pickup_datetime")) / 3600.0),
                2
            )
        ).withColumn(
            # Tip as percentage of fare — our ML signal
            "tip_pct",
            F.round(F.col("tip_amount") / F.col("fare_amount"), 4)
        ).withColumn(
            # Is this trip to/from an airport?
            # JFK=132, LGA=138, EWR=1
            "is_airport_trip",
            F.when(
                F.col("pulocationid").isin([1, 132, 138]) |
                F.col("dolocationid").isin([1, 132, 138]),
                1
            ).otherwise(0)
        ).withColumn(
            # How much does the fare cost per mile?
            "fare_per_mile",
            F.round(F.col("fare_amount") / F.col("trip_distance"), 2)
        ).withColumn(
            # ── ML TARGET ──
            # 1 = generous tipper (tipped more than 20% of fare)
            # 0 = not a generous tipper
            "generous_tipper",
            F.when(F.col("tip_pct") > 0.20, 1)
             .otherwise(0)
             .cast(IntegerType())
        )

    # Filter physically impossible trips AFTER creating duration/speed cols
    # Still no .count() — all lazy
    df = df.filter(
        (F.col("trip_duration_min") >= MIN_DURATION_MIN) &
        (F.col("trip_duration_min") <= MAX_DURATION_MIN) &
        (F.col("avg_speed_mph")     <= MAX_SPEED_MPH)
    )

    log.info("[Step 4] Feature engineering complete ✅")
    log.info("         New features: trip_duration_min, pickup_hour, pickup_dayofweek,")
    log.info("                       is_weekend, time_of_day, avg_speed_mph,")
    log.info("                       tip_pct, is_airport_trip, fare_per_mile, generous_tipper")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Select Final Columns
# ══════════════════════════════════════════════════════════════════════════
def select_final_columns(df: DataFrame) -> DataFrame:
    """Keep only columns needed for ML + auditing. Drop raw timestamps."""

    final_cols = [
        # Audit
        "source_month",
        # Trip info
        "vendorid", "passenger_count", "trip_distance",
        "ratecodeid", "store_and_fwd_flag", "payment_type",
        # Location
        "pulocationid", "dolocationid", "is_airport_trip",
        # Fare breakdown
        "fare_amount", "extra", "mta_tax", "tip_amount",
        "tolls_amount", "improvement_surcharge",
        "congestion_surcharge", "airport_fee", "total_amount",
        # Engineered features
        "trip_duration_min", "pickup_hour", "pickup_dayofweek",
        "is_weekend", "time_of_day", "avg_speed_mph",
        "tip_pct", "fare_per_mile",
        # ML Target
        "generous_tipper",
    ]

    df = df.select(final_cols)
    log.info(f"[Step 5] Final columns selected: {len(final_cols)} ✅")
    return df


# ══════════════════════════════════════════════════════════════════════════
# MAIN TRANSFORM FUNCTION
# ══════════════════════════════════════════════════════════════════════════
def run_transform(spark: SparkSession) -> DataFrame:

    log.info("=" * 55)
    log.info("PHASE 3: TRANSFORM STARTED")
    log.info("=" * 55)

    # ── Extract ────────────────────────────────────────────────────────────
    taxi_df, _ = run_extract(spark)

    # ── Cache ONLY after confirming memory is available ────────────────────
    # DISK_AND_MEMORY: spills to disk if RAM is full instead of crashing
    from pyspark import StorageLevel
    taxi_df.persist(StorageLevel.MEMORY_AND_DISK)

    raw_count = taxi_df.count()  # warms the cache
    log.info(f"Raw input rows (persisted): {raw_count:,}")

    # ── All transform steps (purely lazy — zero Spark actions) ────────────
    df = remove_duplicates(taxi_df)
    df = filter_invalid_rows(df)
    df = handle_nulls(df)
    df = engineer_features(df)
    df = select_final_columns(df)

    # ── Don't count here — return the plan, Load phase will execute it ─────
    log.info("=" * 55)
    log.info("TRANSFORM PLAN BUILT ✅  (execution deferred to Load phase)")
    log.info(f"  Input rows     : {raw_count:,}")
    log.info(f"  Final columns  : {len(df.columns)}")
    log.info(f"  Columns        : {df.columns}")
    log.info("=" * 55)

    return df, raw_count


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Transform") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df, raw_count = run_transform(spark)

    # ── Preview a small sample to verify correctness (NOT a full count) ────
    print("\n📋 TRANSFORMED SCHEMA:")
    df.printSchema()

    print("\n👀 SAMPLE OF TRANSFORMED DATA (5 rows):")
    df.show(5, truncate=True)

    print("\n📊 QUICK STATS (on 100k sample — avoids full scan):")
    sample = df.sample(fraction=0.01, seed=42)   # ~93k rows
    sample.describe(
        "trip_distance", "trip_duration_min",
        "fare_amount", "tip_pct", "avg_speed_mph"
    ).show()

    print("\n🎯 ML TARGET ON SAMPLE:")
    sample.groupBy("generous_tipper").count().show()

    print("\n⏰ TIME OF DAY ON SAMPLE:")
    sample.groupBy("time_of_day") \
          .agg(
              F.count("*").alias("trips"),
              F.round(F.avg("tip_pct"), 4).alias("avg_tip_pct")
          ).orderBy("time_of_day").show()

    spark.stop()
    log.info("Spark session stopped.")


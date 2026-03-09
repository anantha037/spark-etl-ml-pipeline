"""
extract.py
Phase 2 - Extract: Load raw NYC Taxi data from multiple Parquet files
and the Zone lookup CSV into a unified Spark DataFrame.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import LongType, DoubleType

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────
RAW_DATA_PATH  = "/home/anantha/spark-etl-ml-project/data/raw"
TAXI_FILES     = [
    f"{RAW_DATA_PATH}/yellow_tripdata_2023-01.parquet",
    f"{RAW_DATA_PATH}/yellow_tripdata_2023-02.parquet",
    f"{RAW_DATA_PATH}/yellow_tripdata_2023-03.parquet",
]
ZONE_CSV_PATH  = f"{RAW_DATA_PATH}/taxi_zone_lookup.csv"

# Columns we actually need (drop none here — Transform phase will filter)
INT_COLS_TO_CAST = [
    "vendorid", "pulocationid", "dolocationid",
    "payment_type", "ratecodeid", "passenger_count"
]


# ── Helper Functions ───────────────────────────────────────────────────────
def create_spark_session(app_name: str = "NYC_Taxi_ETL") -> SparkSession:
    """Create and return a configured Spark session."""
    log.info("Starting Spark session...")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.shuffle.partitions", "8")   # tuned for local mode
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log.info(f"Spark {spark.version} session started ✅")
    return spark


def normalize_schema(df: DataFrame) -> DataFrame:
    """
    Normalize column names to lowercase and cast integer
    columns to LongType for consistency across monthly files.
    """
    # Lowercase all column names
    df = df.toDF(*[c.lower() for c in df.columns])

    # Cast int/bigint columns to consistent LongType
    for c in INT_COLS_TO_CAST:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(LongType()))

    return df


def read_taxi_parquet(spark: SparkSession, file_paths: list) -> DataFrame:
    """
    Read multiple monthly Parquet files individually,
    normalize each schema, then union into one DataFrame.
    """
    log.info(f"Reading {len(file_paths)} Parquet files...")
    dataframes = []

    for path in file_paths:
        month = path.split("_")[-1].replace(".parquet", "")  # e.g. "2023-01"
        log.info(f"  Loading {month}...")

        df = spark.read \
            .option("spark.sql.parquet.enableVectorizedReader", "false") \
            .parquet(path)

        df = normalize_schema(df)

        # Tag each row with its source month (useful for debugging/auditing)
        df = df.withColumn("source_month", lit(month))

        dataframes.append(df)
        log.info(f"  ✅ {month}: {df.count():,} rows loaded")

    # Find common columns across all files
    common_cols = sorted(
        set(dataframes[0].columns)
        .intersection(*[set(d.columns) for d in dataframes[1:]])
    )
    log.info(f"Common columns across all files: {len(common_cols)}")

    # Union all months using common columns only
    unified_df = dataframes[0].select(common_cols)
    for df in dataframes[1:]:
        unified_df = unified_df.union(df.select(common_cols))

    return unified_df


def read_zone_lookup(spark: SparkSession, path: str) -> DataFrame | None:
    """
    Read the NYC Taxi Zone lookup CSV.
    Returns None if file doesn't exist (it's optional).
    """
    import os
    if not os.path.exists(path):
        log.warning(f"Zone lookup file not found at {path} — skipping.")
        return None

    log.info("Reading Zone lookup CSV...")
    zone_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(path)

    # Normalize column names
    zone_df = zone_df.toDF(*[c.lower() for c in zone_df.columns])
    log.info(f"✅ Zone lookup loaded: {zone_df.count():,} zones")
    return zone_df


def run_extract(spark: SparkSession) -> tuple:
    """
    Main extract function.
    Returns: (taxi_df, zone_df) — zone_df may be None if CSV not present.
    """
    log.info("=" * 55)
    log.info("PHASE 2: EXTRACT STARTED")
    log.info("=" * 55)

    # Extract taxi trip data
    taxi_df = read_taxi_parquet(spark, TAXI_FILES)

    # Extract zone lookup (optional)
    zone_df = read_zone_lookup(spark, ZONE_CSV_PATH)

    # ── Summary Report ─────────────────────────────────────────────────────
    total_rows = taxi_df.count()
    log.info("=" * 55)
    log.info("EXTRACT COMPLETE ✅")
    log.info(f"  Total rows     : {total_rows:,}")
    log.info(f"  Total columns  : {len(taxi_df.columns)}")
    log.info(f"  Zone data      : {'Loaded' if zone_df else 'Not available'}")
    log.info(f"  Months covered : {[f.split('_')[-1].replace('.parquet','') for f in TAXI_FILES]}")
    log.info("=" * 55)

    return taxi_df, zone_df


# ── Entry point (for testing this script standalone) ───────────────────────
if __name__ == "__main__":
    spark = create_spark_session()
    taxi_df, zone_df = run_extract(spark)

    print("\nTAXI SCHEMA:")
    taxi_df.printSchema()

    print("\nSAMPLE ROWS:")
    taxi_df.show(3, truncate=True)

    spark.stop()
    log.info("Spark session stopped.")
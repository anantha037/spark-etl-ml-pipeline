"""
dashboard.py
Phase 13 - Streamlit Dashboard
NYC Taxi Big Data Pipeline — Interactive Dashboard

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Pipeline Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
PREDICTIONS_DIR = os.path.join(BASE, "data", "output", "predictions")
MODELS_PATH   = os.path.join(BASE, "models", "rf_tuned_best")
PIPELINE_PATH = os.path.join(BASE, "models", "feature_pipeline")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2E75B6;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .metric-card.green { border-left-color: #1e7145; }
    .metric-card.orange { border-left-color: #c55a11; }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #1F4E79;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #666;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F4E79;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-green { background: #d4edda; color: #155724; }
    .badge-blue  { background: #cce5ff; color: #004085; }
    .badge-orange{ background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_processed_sample(n=50_000):
    """Load a sample of processed data using pyarrow (no Spark needed)."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    if not os.path.exists(PROCESSED_DIR):
        return None

    all_tables = []
    for month_dir in sorted(os.listdir(PROCESSED_DIR)):
        month_path = os.path.join(PROCESSED_DIR, month_dir)
        if not os.path.isdir(month_path) or not month_dir.startswith("source_month="):
            continue
        for f in os.listdir(month_path):
            if f.endswith(".parquet"):
                table = pq.read_table(os.path.join(month_path, f))
                all_tables.append(table)

    if not all_tables:
        return None

    full = pa.concat_tables(all_tables)
    df = full.to_pandas()

    # Sample if too large
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # Add month label if missing
    if "source_month" not in df.columns:
        df["source_month"] = "Unknown"

    return df


@st.cache_data(show_spinner=False)
def load_predictions_sample(n=50_000):
    """Load a sample of prediction results."""
    import pyarrow.parquet as pq

    if not os.path.exists(PREDICTIONS_DIR):
        return None

    tables = []
    for f in os.listdir(PREDICTIONS_DIR):
        if f.endswith(".parquet"):
            tables.append(pq.read_table(os.path.join(PREDICTIONS_DIR, f)))

    if not tables:
        return None

    import pyarrow as pa
    df = pa.concat_tables(tables).to_pandas()
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return df


def try_spark_predict(features: dict):
    """
    Try real Spark inference. Returns (prediction, confidence, method).
    Falls back to rule-based if Spark unavailable.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.classification import RandomForestClassificationModel
        from pyspark.ml import PipelineModel
        from pyspark.ml.functions import vector_to_array
        import pyspark.sql.functions as F

        spark = SparkSession.builder \
            .appName("TaxiDashboard") \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.sql.parquet.enableVectorizedReader", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        # Build single-row DataFrame
        row = {
            "trip_distance": float(features["trip_distance"]),
            "fare_amount": float(features["fare_amount"]),
            "trip_duration_min": float(features["trip_duration_min"]),
            "pickup_hour": int(features["pickup_hour"]),
            "pickup_dayofweek": int(features["pickup_dayofweek"]),
            "is_weekend": int(features["is_weekend"]),
            "is_airport_trip": int(features["is_airport_trip"]),
            "passenger_count": int(features["passenger_count"]),
            "payment_type": int(features["payment_type"]),
            "pulocationid": int(features["pulocationid"]),
            "dolocationid": int(features["dolocationid"]),
            "avg_speed_mph": float(features["avg_speed_mph"]),
            "ratecodeid": int(features["ratecodeid"]),
            "congestion_surcharge": float(features["congestion_surcharge"]),
            "mta_tax": 0.5,
            "improvement_surcharge": 1.0,
            "extra": 0.5,
            "tolls_amount": 0.0,
            "store_and_fwd_flag": "N",
            "time_of_day": features["time_of_day"],
            "generous_tipper": 0,   # placeholder — not used in prediction
        }

        df = spark.createDataFrame([row])
        pipeline = PipelineModel.load(PIPELINE_PATH)
        df_feat = pipeline.transform(df).select("features", "generous_tipper")
        model = RandomForestClassificationModel.load(MODELS_PATH)
        pred_df = model.transform(df_feat)
        result = pred_df.withColumn(
            "confidence", vector_to_array(F.col("probability"))[1]
        ).select("prediction", "confidence").collect()[0]

        return int(result["prediction"]), float(result["confidence"]), "Spark ML Model"

    except Exception:
        return fallback_predict(features)


def fallback_predict(features: dict):
    """
    Rule-based fallback when Spark is not available.
    Mimics the Random Forest logic using the key signals from feature importance.
    """
    # Payment type is 93.4% of importance
    # 1 = Credit Card, 2 = Cash
    payment = int(features["payment_type"])
    distance = float(features["trip_distance"])
    fare = float(features["fare_amount"])
    is_airport = int(features["is_airport_trip"])
    hour = int(features["pickup_hour"])

    if payment == 2:  # Cash — almost never tips electronically
        confidence = 0.06
        prediction = 0
    elif payment == 1:  # Credit card
        base = 0.72
        # Airport trips tip better
        if is_airport:
            base += 0.08
        # Evening/night tip better
        if hour in range(18, 24) or hour in range(0, 6):
            base += 0.04
        # Longer trips tip slightly better
        if distance > 5:
            base += 0.03
        # Higher fares
        if fare > 20:
            base += 0.03
        confidence = min(base, 0.97)
        prediction = 1 if confidence >= 0.5 else 0
    else:
        confidence = 0.5
        prediction = 1

    return prediction, round(confidence, 4), "Rule-based fallback (Spark unavailable)"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚕 NYC Taxi Pipeline")
    st.markdown("**Apache Spark ETL + ML**")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("NYC TLC Yellow Taxi 2023")
    st.markdown("Jan · Feb · Mar")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("""
    <span class='badge badge-blue'>Spark 3.5.1</span>
    <span class='badge badge-green'>PySpark</span><br><br>
    <span class='badge badge-orange'>PostgreSQL</span>
    <span class='badge badge-blue'>Parquet</span>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by Anantha · March 2026")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Trip Explorer",
    "🔮  Tip Predictor",
    "📈  Pipeline Stats"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRIP EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Trip Explorer")
    st.caption("Exploring 50K sampled trips from the cleaned dataset (8.9M total)")

    with st.spinner("Loading trip data..."):
        df = load_processed_sample()

    if df is None:
        st.warning("Processed data not found. Run the ETL pipeline first: `python src/pipeline/etl_pipeline.py`")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    months = sorted(df["source_month"].unique()) if "source_month" in df.columns else ["All"]
    with col1:
        selected_months = st.multiselect("Month", months, default=months)
    with col2:
        time_opts = sorted(df["time_of_day"].dropna().unique()) if "time_of_day" in df.columns else []
        selected_times = st.multiselect("Time of Day", time_opts, default=time_opts)
    with col3:
        airport_filter = st.selectbox("Trip Type", ["All", "Airport Trips", "Non-Airport"])
    with col4:
        payment_map = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute"}
        pay_opts = ["All"] + [payment_map.get(p, str(p)) for p in sorted(df["payment_type"].dropna().unique())]
        selected_payment = st.selectbox("Payment Type", pay_opts)

    # Apply filters
    fdf = df.copy()
    if selected_months:
        fdf = fdf[fdf["source_month"].isin(selected_months)]
    if selected_times:
        fdf = fdf[fdf["time_of_day"].isin(selected_times)]
    if airport_filter == "Airport Trips":
        fdf = fdf[fdf["is_airport_trip"] == 1]
    elif airport_filter == "Non-Airport":
        fdf = fdf[fdf["is_airport_trip"] == 0]
    if selected_payment != "All":
        rev_map = {v: k for k, v in payment_map.items()}
        pay_code = rev_map.get(selected_payment)
        if pay_code:
            fdf = fdf[fdf["payment_type"] == pay_code]

    st.caption(f"Showing **{len(fdf):,}** trips after filters")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    avg_fare    = fdf["fare_amount"].mean() if "fare_amount" in fdf else 0
    avg_tip     = fdf["tip_amount"].mean() if "tip_amount" in fdf else 0
    avg_dist    = fdf["trip_distance"].mean() if "trip_distance" in fdf else 0
    avg_dur     = fdf["trip_duration_min"].mean() if "trip_duration_min" in fdf else 0
    pct_generous= fdf["generous_tipper"].mean() * 100 if "generous_tipper" in fdf else 0

    k1.metric("Avg Fare", f"${avg_fare:.2f}")
    k2.metric("Avg Tip", f"${avg_tip:.2f}")
    k3.metric("Avg Distance", f"{avg_dist:.1f} mi")
    k4.metric("Avg Duration", f"{avg_dur:.0f} min")
    k5.metric("Generous Tippers", f"{pct_generous:.1f}%")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-header">Trips by Hour of Day</div>', unsafe_allow_html=True)
        if "pickup_hour" in fdf.columns:
            hourly = fdf.groupby("pickup_hour").size().reset_index(name="trips")
            st.bar_chart(hourly.set_index("pickup_hour")["trips"])

    with ch2:
        st.markdown('<div class="section-header">Fare Distribution</div>', unsafe_allow_html=True)
        if "fare_amount" in fdf.columns:
            fare_data = fdf["fare_amount"].clip(0, 80)
            hist_vals, hist_bins = np.histogram(fare_data, bins=40)
            hist_df = pd.DataFrame({
                "fare_bucket": [f"${b:.0f}" for b in hist_bins[:-1]],
                "count": hist_vals
            })
            st.bar_chart(hist_df.set_index("fare_bucket")["count"])

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown('<div class="section-header">Avg Fare by Time of Day</div>', unsafe_allow_html=True)
        if "time_of_day" in fdf.columns and "fare_amount" in fdf.columns:
            tod = fdf.groupby("time_of_day")["fare_amount"].mean().round(2)
            st.bar_chart(tod)

    with ch4:
        st.markdown('<div class="section-header">Generous Tippers by Payment Type</div>', unsafe_allow_html=True)
        if "payment_type" in fdf.columns and "generous_tipper" in fdf.columns:
            pt = fdf.groupby("payment_type")["generous_tipper"].mean() * 100
            pt.index = [payment_map.get(i, str(i)) for i in pt.index]
            st.bar_chart(pt.round(1))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — TIP PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔮 Tip Predictor")
    st.caption("Enter trip details — the trained Random Forest model predicts whether this passenger will tip generously (>20%).")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-header">Trip Details</div>', unsafe_allow_html=True)

        payment_type = st.selectbox(
            "Payment Type",
            options=[1, 2, 3],
            format_func=lambda x: {1: "💳 Credit Card", 2: "💵 Cash", 3: "📋 No Charge"}[x]
        )

        c1, c2 = st.columns(2)
        with c1:
            trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 3.5, 0.1)
            fare_amount   = st.slider("Fare Amount ($)", 2.5, 100.0, 15.0, 0.5)
            passenger_count = st.selectbox("Passengers", [1, 2, 3, 4, 5, 6], index=0)
        with c2:
            pickup_hour   = st.slider("Pickup Hour (0–23)", 0, 23, 18)
            trip_duration = st.slider("Duration (minutes)", 1, 120, 20)
            ratecodeid    = st.selectbox("Rate Code",
                                          options=[1, 2, 3, 4, 5, 6],
                                          format_func=lambda x: {1:"Standard",2:"JFK",3:"Newark",4:"Nassau/Westchester",5:"Negotiated",6:"Group"}[x])

        c3, c4 = st.columns(2)
        with c3:
            is_airport    = st.checkbox("Airport Trip (JFK/LGA/EWR)", value=False)
        with c4:
            day_of_week   = st.selectbox("Day of Week",
                                          options=[1,2,3,4,5,6,7],
                                          format_func=lambda x: {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}[x])

        # Derived fields
        is_weekend = 1 if day_of_week in [1, 7] else 0
        avg_speed  = trip_distance / (trip_duration / 60) if trip_duration > 0 else 0
        if pickup_hour in range(0, 6):
            time_of_day = "night"
        elif pickup_hour in range(6, 12):
            time_of_day = "morning"
        elif pickup_hour in range(12, 18):
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        features = {
            "trip_distance": trip_distance,
            "fare_amount": fare_amount,
            "trip_duration_min": trip_duration,
            "pickup_hour": pickup_hour,
            "pickup_dayofweek": day_of_week,
            "is_weekend": is_weekend,
            "is_airport_trip": int(is_airport),
            "passenger_count": passenger_count,
            "payment_type": payment_type,
            "pulocationid": 161,  # Midtown default
            "dolocationid": 132,
            "avg_speed_mph": round(avg_speed, 2),
            "ratecodeid": ratecodeid,
            "congestion_surcharge": 2.5,
            "time_of_day": time_of_day,
        }

        predict_btn = st.button("🔮 Predict Tip Behaviour", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Running inference..."):
                t0 = time.time()
                prediction, confidence, method = try_spark_predict(features)
                elapsed = round(time.time() - t0, 2)

            if prediction == 1:
                st.success("✅ **GENEROUS TIPPER** predicted")
                st.markdown(f"""
                <div class="metric-card green">
                    <div class="metric-val">{confidence*100:.1f}%</div>
                    <div class="metric-label">Model confidence (class 1)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ **NOT a generous tipper** predicted")
                st.markdown(f"""
                <div class="metric-card orange">
                    <div class="metric-val">{(1-confidence)*100:.1f}%</div>
                    <div class="metric-label">Confidence (not generous)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Trip Summary**")
            st.markdown(f"""
            | Feature | Value |
            |---------|-------|
            | Distance | {trip_distance} miles |
            | Fare | ${fare_amount:.2f} |
            | Duration | {trip_duration} min |
            | Speed | {avg_speed:.1f} mph |
            | Payment | {'Credit Card' if payment_type==1 else 'Cash' if payment_type==2 else 'Other'} |
            | Pickup Hour | {pickup_hour}:00 ({time_of_day}) |
            | Airport Trip | {'Yes' if is_airport else 'No'} |
            | Weekend | {'Yes' if is_weekend else 'No'} |
            """)

            st.caption(f"⚙️ Method: *{method}* · {elapsed}s")

            if method != "Spark ML Model":
                st.info("💡 For real model inference, make sure Spark is installed and models are present at `models/rf_tuned_best/`")

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#999;'>
                <div style='font-size:3rem'>🚕</div>
                <div style='margin-top:12px;'>Fill in the trip details on the left<br>and click <b>Predict</b></div>
                <br>
                <div style='font-size:0.8rem;'>
                    Uses the tuned Random Forest model<br>
                    trained on 7.1 million NYC taxi trips
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show a quick note about what matters most
            st.markdown("---")
            st.markdown("**What the model learned:**")
            importance_data = pd.DataFrame({
                "Feature": ["Payment Type", "Rate Code ID", "Congestion Surcharge",
                             "Fare Amount", "Trip Distance"],
                "Importance": [93.4, 1.7, 1.5, 1.0, 0.8]
            })
            st.bar_chart(importance_data.set_index("Feature")["Importance"])
            st.caption("Payment type dominates — card vs cash is the strongest signal")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE STATS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Pipeline Stats")
    st.caption("End-to-end results from the full pipeline run")

    # ── ETL Metrics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ETL Pipeline Results</div>', unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Raw Rows", "9,384,487")
    e2.metric("Clean Rows", "8,934,307")
    e3.metric("Rows Removed", "450,180", delta="-4.8%", delta_color="off")
    e4.metric("ETL Runtime", "79s")
    e5.metric("Partitions", "3 months")

    monthly_data = pd.DataFrame({
        "Month": ["2023-01", "2023-02", "2023-03"],
        "Raw Rows": [3_066_766, 2_913_955, 3_403_766],
        "Clean Rows": [2_925_717, 2_770_759, 3_237_831],
    })
    monthly_data["Removed"] = monthly_data["Raw Rows"] - monthly_data["Clean Rows"]
    st.dataframe(monthly_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── ML Metrics ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ML Model Results</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "80.0%")
    m2.metric("AUC-ROC", "0.789")
    m3.metric("F1 Score", "0.778")
    m4.metric("Precision", "0.849")
    m5.metric("Recall", "0.800")

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Random Forest — Confusion Matrix**")
        cm_df = pd.DataFrame({
            "": ["Actual: Not Generous", "Actual: Generous"],
            "Predicted: Not Generous": ["327,423 (TN)", "59 (FN)"],
            "Predicted: Generous": ["358,154 (FP)", "1,102,281 (TP)"]
        })
        st.dataframe(cm_df.set_index(""), use_container_width=True)
        st.caption("FN = 59 out of 1.1M genuine generous tippers — near-perfect recall for class 1")

    with mc2:
        st.markdown("**Linear Regression (Fare Prediction)**")
        lr_df = pd.DataFrame({
            "Metric": ["R² Score", "RMSE", "MAE", "Training Time"],
            "Value": ["0.943", "$4.04", "$1.38", "59.8 seconds"]
        })
        st.dataframe(lr_df.set_index("Metric"), use_container_width=True)
        st.caption("Model explains 94.3% of fare variance — trip distance is the strongest predictor")

    st.markdown("---")

    # ── Optimization Benchmarks ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Optimization Benchmarks</div>', unsafe_allow_html=True)

    bench_df = pd.DataFrame({
        "Technique": ["Column Pruning", "Partition Pruning", "AQE", "Caching (3 ops)"],
        "Without (s)": [37.8, 4.5, 7.8, 16.2],
        "With (s)": [6.3, 2.1, 5.5, 59.4],
        "Speedup": ["6.0x 🏆", "2.1x ✅", "1.4x ✅", "0.3x ⚠️"]
    })
    st.dataframe(bench_df.set_index("Technique"), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        st.bar_chart(bench_df.set_index("Technique")[["Without (s)", "With (s)"]])
    with b2:
        st.markdown("**Key Insight**")
        st.markdown("""
        - **Column pruning wins** because Parquet is columnar — unused columns are never read from disk
        - **Caching was slower** here because data is on SSD and only 3 operations ran. Caching pays off with 10+ operations
        - **Partition pruning** scales massively in production with hundreds of monthly partitions
        - **AQE** provides larger gains on skewed data and complex joins
        """)

    st.markdown("---")

    # ── Full Pipeline Timing ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Pipeline Execution</div>', unsafe_allow_html=True)
    timing_df = pd.DataFrame({
        "Stage": ["ETL", "ML Prep", "Predict", "Save"],
        "Duration (s)": [79.6, 9.2, 33.5, 54.5]
    })
    t1, t2 = st.columns([1, 2])
    with t1:
        st.dataframe(timing_df.set_index("Stage"), use_container_width=True)
        st.metric("Total", "181.5s (3.0 min)")
    with t2:
        st.bar_chart(timing_df.set_index("Stage")["Duration (s)"])

    st.markdown("---")

    # ── Prediction Output Sample ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Predictions (from saved output)</div>', unsafe_allow_html=True)
    with st.spinner("Loading prediction sample..."):
        pred_df = load_predictions_sample(n=5000)

    if pred_df is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Predictions", "1,787,917")
        p2.metric("Predicted Generous", f"{pred_df['prediction'].mean()*100:.1f}%")
        p3.metric("Avg Confidence", f"{pred_df['confidence'].mean():.3f}")

        conf_hist = np.histogram(pred_df["confidence"], bins=30)
        conf_df = pd.DataFrame({
            "confidence_bucket": [f"{b:.2f}" for b in conf_hist[1][:-1]],
            "count": conf_hist[0]
        })
        st.markdown("**Confidence Score Distribution**")
        st.bar_chart(conf_df.set_index("confidence_bucket")["count"])
        st.caption("Two peaks: near 0.06 (cash passengers, low confidence) and near 0.80 (card passengers, high confidence)")
    else:
        st.info("Prediction output not found. Run `python src/pipeline/full_pipeline.py` to generate predictions.")

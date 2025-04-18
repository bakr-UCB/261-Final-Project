# features.py
from databricks.sdk.runtime import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from graphframes import GraphFrame
from typing import Optional, List
from pathlib import Path
import math
from functools import reduce
import os

# Prophet and Pandas in driver mode
from prophet import Prophet
import pandas as pd
import holidays

def add_local_time_features(
    df: DataFrame,
    time_col: str = "sched_depart_date_time",
    test_mode: bool = False
) -> DataFrame:
    """
    Add local time-based features with Fourier transforms.

    Args:
        df: Spark DataFrame with local datetime column
        time_col: Column name with local datetime
        test_mode: Enable logs for debugging

    Returns:
        DataFrame with new time features
    """
    if test_mode:
        print("🔁 add_local_time_features...")

    df = df.withColumn("local_hour", hour(col(time_col)))
    df = df.withColumn("local_dow", dayofweek(col(time_col)))
    df = df.withColumn("local_month", month(col(time_col)))

    df = df.withColumn("sin_hour", sin(2 * lit(math.pi) * col("local_hour") / 24))
    df = df.withColumn("cos_hour", cos(2 * lit(math.pi) * col("local_hour") / 24))
    df = df.withColumn("sin_dow", sin(2 * lit(math.pi) * col("local_dow") / 7))
    df = df.withColumn("cos_dow", cos(2 * lit(math.pi) * col("local_dow") / 7))

    return df


def compute_and_join_pagerank_metrics(
    df: DataFrame,
    base_path: str,
    year_col: str = "YEAR",
    quarter_col: str = "QUARTER",
    test_mode: bool = False
) -> DataFrame:
    """
    Compute and join previous year's PageRank and degree metrics.

    Args:
        df: Input flights DataFrame
        base_path: Root path for storing metrics
        year_col: Column for extracting year
        test_mode: If True, restrict to 1 year

    Returns:
        DataFrame with graph metrics joined
    """
    print("🔁 compute_and_join_pagerank_metrics...")

    def prev_period(year: int, quarter: int) -> (int, int):
        if quarter == 1:
            return (year - 1, 4)
        else:
            return (year, quarter - 1)

    if test_mode:
        pairs = [(2016, 1)]
    else:
        pairs = df.select(year_col, quarter_col).distinct().orderBy(year_col, quarter_col).rdd.map(lambda r: (r[0], r[1])).collect()

    # years = [2016] if test_mode else [r[0] for r in df.select(year_col).distinct().collect()]
    output_dfs = []

    # # for year in years:
    #     print(f"📆 Processing year {year}...")

    #     this_year_df = df.filter(col(year_col) == year)
    #     last_year_df = df.filter(col(year_col) == year - 1)

    #     save_path = f"{base_path}/airport_pagerank/year={year - 1}/"
    #     if Path(save_path).exists():
    #         pr_df = spark.read.parquet(save_path)
    #         print(f"✅ Loaded cached graph metrics for {year - 1}")

    for year, quarter in pairs:
        print(f"📆 Processing {year} Q{quarter}...")
        prev_year, prev_quarter = prev_period(year, quarter)

        this_period_df = df.filter((col(year_col) == year) & (col(quarter_col) == quarter))
        last_period_df = df.filter((col(year_col) == prev_year) & (col(quarter_col) == prev_quarter))

        save_path = f"{base_path}/airport_pagerank/year{prev_year}_q{prev_quarter}/"

        if Path(save_path).exists():
            pr_df = spark.read.parquet(save_path)
            print(f"✅ Loaded cached graph metrics for {prev_year} {prev_quarter}")

        else:
            edges = last_period_df.groupBy("ORIGIN", "DEST").count().withColumnRenamed("count", "weight")
            vertices = edges.selectExpr("ORIGIN as id").union(edges.selectExpr("DEST as id")).distinct()
            g = GraphFrame(vertices, edges.withColumnRenamed("ORIGIN", "src").withColumnRenamed("DEST", "dst"))

            pr_df = g.pageRank(resetProbability=0.15, maxIter=10).vertices
            deg_df = g.inDegrees.withColumnRenamed("inDegree", "in_degree") \
                .join(g.outDegrees.withColumnRenamed("outDegree", "out_degree"), "id", "outer")
            pr_df = pr_df.join(deg_df, "id", "left")

            pr_df.write.mode("overwrite").parquet(save_path)
            print(f"💾 Saved PageRank for {prev_year} {prev_quarter}")

        joined = this_period_df.join(pr_df.withColumnRenamed("id", "ORIGIN"), on="ORIGIN", how="left")
        output_dfs.append(joined)

    return output_dfs[0] if test_mode else reduce(DataFrame.unionByName, output_dfs)


def generate_lagged_delay_aggregates(
    df: DataFrame,
    base_path: str,
    test_mode: bool = False
) -> DataFrame:
    """
    Compute rolling 30-day delay aggregates by ORIGIN.

    Args:
        df: Flights DataFrame
        base_path: Where to save aggregated features
        test_mode: If True, restrict to 2016

    Returns:
        DataFrame with delay aggregates
    """
    print("🔁 generate_lagged_delay_aggregates...")

    df = df.withColumn("dep_date", to_date("FL_DATE"))
    if test_mode:
        print("🧪 Using test mode: only 2016")
        df = df.filter(col("YEAR") == 2016)

    agg = df.groupBy("ORIGIN", "dep_date") \
        .agg(
            avg("DEP_DELAY").alias("mean_dep_delay"),
            avg((col("outcome") > 0).cast("double")).alias("prop_delayed")
        )

    win = Window.partitionBy("ORIGIN").orderBy("dep_date").rowsBetween(-30, -1)
    agg = agg.withColumn("mean_dep_delay_30d", avg("mean_dep_delay").over(win))
    agg = agg.withColumn("prop_delayed_30d", avg("prop_delayed").over(win))

    output_path = f"{base_path}/delay_aggregates/"
    agg.write.mode("overwrite").parquet(output_path)
    print(f"💾 Delay aggregates saved to {output_path}")

    return df.join(agg, on=["ORIGIN", "dep_date"], how="left")


def add_prophet_features_per_airport(
    spark_df: DataFrame,
    base_path: str,
    test_mode: bool = False
) -> DataFrame:
    """
    Adds Prophet-based seasonality/trend features per airport.
    Ensures features are based only on past dates (no leakage).

    Args:
        spark_df: Input DataFrame with at least FL_DATE, ORIGIN, and outcome
        base_path: Path to save/load cached Prophet outputs
        test_mode: Limit to 1-2 airports for dev

    Returns:
        DataFrame with trend, seasonal, and is_holiday_week features
    """
    print("🔁 add_prophet_features_per_airport...")

    from pyspark.sql.functions import to_date
    airport_list = ["JFK", "ORD"] if test_mode else spark_df.select("ORIGIN").distinct().rdd.map(lambda r: r[0]).collect()

    result_frames = []

    for airport in airport_list:
        print(f"🔮 Prophet for airport: {airport}")
        save_path = f"{base_path}/prophet_outputs/airport={airport}/"

        if Path(save_path).exists():
            prophet_df = spark.read.parquet(save_path)
            print(f"✅ Loaded cached Prophet features for {airport}")
        else:
            sdf = spark_df.filter(col("ORIGIN") == airport).withColumn("ds", to_date("FL_DATE"))
            df_pd = sdf.groupBy("ds").agg(avg("outcome").alias("y")).orderBy("ds").toPandas()

            if df_pd.shape[0] < 90:
                print(f"⚠️ Not enough data for Prophet at {airport} — skipping.")
                continue
            df_pd["ds"] = pd.to_datetime(df_pd["ds"])
            us_holidays = holidays.US(years=range(2014, 2020))
            holiday_dates = [pd.Timestamp(h) for h in list(us_holidays.keys())]



            df_pd["holidays"] = df_pd["ds"].apply(lambda d: [__builtins__.abs((d - h).days) for h in us_holidays_ts])            
            df_pd["is_holiday_week"] = df_pd["ds"].apply(lambda d: any(__builtins__.abs((d - pd.Timestamp(h)).days) <= 3 for h in us_holidays_ts))

            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=True,
                holidays_prior_scale=10
            )
            model.add_country_holidays(country_name='US')
            model.fit(df_pd)

            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            merged = pd.merge(forecast, df_pd, how="left", on="ds")
            merged["residual"] = merged["y"] - merged["yhat"]
            merged["FL_DATE"] = merged["ds"] + pd.Timedelta(days=7)
            # Extract only past dates & components
            prophet_features = merged[["trend", "FL_DATE", "weekly", "daily", "yearly", "residual", "additive_terms", "multiplicative_terms", "is_holiday_week"]].copy()
            prophet_features["ORIGIN"] = airport
            prophet_features["is_holiday_week"] = df_pd.set_index("ds")["is_holiday_week"].reindex(forecast["ds"]).fillna(False).values

            final = spark.createDataFrame(prophet_features.rename(columns={"ds": "FL_DATE"}))
            final.write.mode("overwrite").parquet(save_path)
            print(f"💾 Saved Prophet output for {airport}")
            prophet_df = final

        result_frames.append(prophet_df)

    if not result_frames:
        print("⚠️ No Prophet data generated.")
        return spark_df

    # Join with flight data on ORIGIN + FL_DATE (safe to use for pre-departure)
    all_prophet = reduce(DataFrame.unionByName, result_frames)
    return spark_df.join(all_prophet, on=["ORIGIN", "FL_DATE"], how="left")


# I/O utilities
def save_features(df: DataFrame, path: str) -> None:
    df.write.mode("overwrite").parquet(path)

def load_features_if_exists(path: str) -> Optional[DataFrame]:
    return spark.read.parquet(path) if Path(path).exists() else None

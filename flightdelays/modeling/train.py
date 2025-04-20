from typing import List, Dict, Tuple, Any, Union,Callable
import numpy as np
import random
from datetime import datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics,BinaryClassificationMetrics
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    MultilayerPerceptronClassifier
)
from xgboost.spark import SparkXGBClassifier
import mlflow
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials


def train_test_split_timeseries(df: DataFrame, time_col: str, split_method: str= "frac", test_start: str= None, test_fraction: float = 0.2, max_date: str = "2100-01-01", verbose: bool = True) -> tuple[DataFrame, DataFrame]:
    """
    Splits a PySpark DataFrame into a train/test set based on a timestamp column.
    The most recent `test_fraction` of the data (by time) is used as test set.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        time_col (str): Timestamp column name (must be sortable).
        split_method (str): Method to split the data. Can be "frac" or "date".
        split_date (str): Date to split the data. Used if `split_method` is "date".
        test_fraction (float): Fraction of time span to allocate to the test set.
        max_date (str): Maximum date to consider (default: "2100-01-01").
        verbose (bool): Print boundaries and sizes.

    Returns:
        (train_df, test_df): Tuple of train and test DataFrames.
    """
    # Filter df to before max date
    df = df.filter(F.col(time_col) < max_date)

    # Get min and max time
    min_time, max_time = df.selectExpr(f"min({time_col})", f"max({time_col})").first()
    total_days = (max_time - min_time).days
    if test_start is not None:
        test_start = datetime.strptime(test_start, "%Y-%m-%d")

    if split_method == "frac":
        test_days = int(total_days * test_fraction)
        test_start = max_time - timedelta(days=test_days)

    train_df = df.filter(F.col(time_col) < test_start)
    test_df = df.filter(F.col(time_col) >= test_start)

    if verbose:
        print(f"📅 Total date range: {min_time.date()} → {max_time.date()} ({total_days} days)")
        print(f"✅ Train: {min_time.date()} → {test_start.date()} ({train_df.count():,} rows)")
        print(f"🧪 Test: {test_start.date()} → {max_time.date()} ({test_df.count():,} rows)")

    return train_df, test_df

def time_series_cv_folds(
    df: DataFrame,
    time_col: str,
    k: int=3,
    blocking: bool=False,
    overlap: float=0.0,
    verbose: bool=False
) -> list[Tuple[DataFrame, DataFrame]]:
    """
    Split a time-series PySpark DataFrame into k train/test folds with optional overlap and blocking.
    
    Args:
        df (DataFrame): PySpark DataFrame with a timestamp column.
        dep_utc_time_colvarname (str): Name of the timestamp column.
        k (int): Number of folds.
        blocking (bool): Whether to block the training set to avoid cumulative data.
        overlap (float): Fraction of overlap between validation windows (e.g. 0.2 = 20% overlap).
        verbose (bool): Whether to print the time splits.
        
    Returns:
        List of (train_df, val_df) tuples.
    """
    # Get time boundaries
    min_date = df.select(F.min(time_col)).first()[0]
    max_date = df.select(F.max(time_col)).first()[0]
    n_days = (max_date - min_date).days + 1

    # Adjust chunk sizing
    total_width = k + 1 - overlap * (k - 1)
    chunk_size = int(np.ceil(n_days / total_width))

    if verbose:
        print(f"Splitting data into {k} folds with {overlap*100:.0f}% overlap")
        print(f"Min date: {min_date}, Max date: {max_date}")
        print(f"{chunk_size:,} days per fold")
        print("************************************************************")

    folds = []
    for i in range(k):
        # Offset calculation with overlap
        train_start_offset = 0 if not blocking else int(i * (1 - overlap) * chunk_size)
        train_end_offset = int((i + 1) * chunk_size)
        val_start_offset = train_end_offset
        val_end_offset = int(val_start_offset + chunk_size)

        # Compute actual timestamps
        train_start = min_date + timedelta(days=train_start_offset)
        train_end = min_date + timedelta(days=train_end_offset)
        val_start = min_date + timedelta(days=val_start_offset)
        val_end = min_date + timedelta(days=val_end_offset)

        if val_start >= max_date:
            break
        if val_end > max_date:
            val_end = max_date + timedelta(days=1)

        # Apply filters
        train_df = df.filter((F.col(time_col) >= train_start) & (F.col(time_col) < train_end))
        val_df = df.filter((F.col(time_col) >= val_start) & (F.col(time_col) < val_end))

        if verbose:
            print(f"Fold {i + 1}:")
            print(f"  TRAIN: {train_start.date()} → {train_end.date()} ({train_df.count():,} rows)")
            print(f"  VAL:   {val_start.date()} → {val_end.date()} ({val_df.count():,} rows)")
            print("------------------------------------------------------------")

        folds.append((train_df, val_df))

    return folds


def add_class_weights(df, label_col: str):
    label_counts = df.groupBy(label_col).count().toPandas()
    neg, pos = label_counts.sort_values(label_col)["count"].tolist()
    pos_weight = float(neg) / pos

    df_weighted = df.withColumn("weight", F.when(F.col(label_col) == 1, pos_weight).otherwise(1.0))
    return df_weighted, pos_weight


def downsample(train_df: DataFrame,verbose: bool=False) -> DataFrame:
    '''
    Downsamples train_df to balance classes
    Input: train_df
    Output: train_df
    '''
    #balance classes in train
    delay_count = train_df.filter(F.col("outcome") == 1).count()
    non_delay_count = train_df.filter(F.col("outcome") == 0).count()

    total = delay_count + non_delay_count
    keep_percent = delay_count / non_delay_count

    train_delay = train_df.filter(F.col('outcome') == 1)
    train_non_delay = train_df.filter(F.col('outcome') == 0).sample(withReplacement=False,fraction=keep_percent,seed=42)
    train_downsampled = train_delay.union(train_non_delay)
    return train_downsampled


def cv_eval(predictions: DataFrame, label_col="outcome", prediction_col="prediction", metric:str="F2"):
    """
    Input: transformed df with prediction and label
    Output: desired score 
    """
    rdd_preds_m = predictions.select(['prediction', label_col]).rdd
    rdd_preds_b = predictions.select('outcome','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['outcome'])))
    metrics_m = MulticlassMetrics(rdd_preds_m)
    metrics_b = BinaryClassificationMetrics(rdd_preds_b)
    if metric == "F2":
        score = np.round(metrics_m.fMeasure(label=1.0, beta=2.0), 4)
    elif metric == "pr":
        score = metrics_b.areaUnderPR
    return score


def model_tuner(
    model_name: str,
    model_params: Dict[str, Any],
    stages,
    folds: List[Tuple[DataFrame, DataFrame]],
    features,
    label: str = "outcome",
    mlflow_run_name: str = "/Users/m.bakr@berkeley.edu/flight_delay_tuning",
    metric: str = "F2",
    verbose: bool = True
) -> Dict[str, Union[float, str, Dict[str, Any]]]:
    """
    Universal tuning function for PySpark classification models using time-series cross-validation.

    Args:
        model_name (str): One of ['logreg', 'rf', 'mlp', 'xgb']
        model_params (Dict[str, Any]): Parameters to apply to the model
        folds (List of (train_df, val_df)): Time-aware CV folds
        mlflow_run_name (str): Optional MLflow parent run name
        verbose (bool): Whether to log outputs during tuning

    Returns:
        Dict with best average F2 or pr score, model name, and parameters
    """

    # Model factory
    model_factory = {
        "logreg": LogisticRegression,
        "rf": RandomForestClassifier,
        "mlp": MultilayerPerceptronClassifier,
        "xgb": SparkXGBClassifier
    }

    assert model_name in model_factory, f"Unsupported model: {model_name}"

    ModelClass = model_factory[model_name]

    # Apply required fields
    model = ModelClass(
        featuresCol=features,
        labelCol=label,
        # weightCol="weight",  # Handles imbalance
        **model_params
    )

    pipeline = Pipeline(stages=[model] + stages) 

    scores = []

    with mlflow.start_run(run_name=mlflow_run_name):
        for i, (train_df, val_df) in enumerate(folds):
            fitted_model = pipeline.fit(train_df)
            preds = fitted_model.transform(val_df)
            score = cv_eval(preds, metric)
            scores.append(score)

            if verbose:
                print(f"[Fold {i+1}] {metric} Score: {score:.4f}")

            mlflow.log_metric(f"{metric}_fold_{i+1}", score)

        avg_score = float(np.mean(scores))
        mlflow.log_param("model", model_name)
        mlflow.log_params(model_params)
        mlflow.log_metric("avg_{metric}_score", avg_score)

        if verbose:
            print(f"✅ Average {metric} Score: {avg_score:.4f} | Model: {model_name}")

    return {
        "model": model_name,
        "params": model_params,
        "avg_f2_score": avg_score
    }


def make_hyperopt_objective(
    model_name: str,
    folds: List[Tuple[DataFrame, DataFrame]],
    stages: List,
    param_space_converter: Callable[[Dict[str, Any]], Dict[str, Any]],
    features,
    label: str = "outcome",
    metric: str = "F2",
    mlflow_experiment_name: str = "Hyperopt_Universal_Tuning",
    verbose: bool = True
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates a Hyperopt-compatible objective function for any PySpark classifier.

    Args:
        model_name (str): One of 'logreg', 'rf', 'mlp', 'xgb'.
        folds (List of (train_df, val_df)): Time-series CV folds.
        param_space_converter (Callable): Converts Hyperopt sample into model params.
        mlflow_experiment_name (str): MLflow experiment name.
        verbose (bool): Logging toggle.

    Returns:
        Callable that can be passed as fn to hyperopt.fmin()
    """

    def objective(sampled_params: Dict[str, Any]) -> Dict[str, Any]:
        # Convert sampled param space to Spark-friendly params
        model_params = param_space_converter(sampled_params)

        result = model_tuner(
            model_name=model_name,
            model_params=model_params,
            stages=stages,
            folds=folds,
            features=features,
            label=label,
            metric=metric,
            mlflow_run_name=f"hyperopt_{model_name}",
            verbose=verbose
        )
        print(f"result: {result}, result type: {type(result)}")

        return {
            "loss": -result[1],  # Minimize negative F2
            "status": STATUS_OK,
            "params": result[2
                             ]
        }

    return objective

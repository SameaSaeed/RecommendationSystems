from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from src.utilities.config import config

def combine_predictions_and_evaluate(
    spark: SparkSession,
    predictions: DataFrame,
    item_cf_predictions: DataFrame,
    test_df: DataFrame,
    rmse: float,
    output_path: str = config.output_path
):
    """
    Combine ALS and Item CF predictions, calculate hybrid predictions, evaluate RMSE, and save metrics.

    Parameters
    ----------
    spark : SparkSession
    predictions : DataFrame
        ALS predictions with columns ['userId', 'movieId', 'prediction']
    item_cf_predictions : DataFrame
        Item CF predictions with columns ['userId', 'target_movieId', 'prediction']
    test_df : DataFrame
        Test ratings with columns ['userId', 'movieId', 'rating']
    rmse : float
        ALS RMSE
    output_path : str
        Base path to save predictions and metrics
    """

    # Rename prediction columns
    als_predictions = predictions.withColumnRenamed("prediction", "als_prediction")
    item_cf_predictions = item_cf_predictions.withColumnRenamed("prediction", "cf_prediction") \
                                             .withColumnRenamed("target_movieId", "movieId")

    # Join ALS and CF
    combined_predictions = als_predictions.join(item_cf_predictions, ["userId", "movieId"], "inner")

    # Initial hybrid: simple average
    combined_predictions = combined_predictions.withColumn(
        "hybrid_prediction",
        (F.col("als_prediction") + F.col("cf_prediction")) / 2
    )

    # Save hybrid predictions
    combined_predictions.write.mode("overwrite").parquet(f"{output_path}/hybrid_predictions")

    # Prepare test dataset
    test_renamed = test_df.withColumnRenamed("rating", "true_rating")

    # Evaluate hybrid RMSE
    evaluation_df = combined_predictions.join(test_renamed, ["userId", "movieId"], "inner") \
                                       .filter(
                                           F.col("true_rating").isNotNull() &
                                           F.col("hybrid_prediction").isNotNull()
                                       )

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="true_rating", predictionCol="hybrid_prediction")
    hybrid_rmse = evaluator.evaluate(evaluation_df)

    print(f"Hybrid Model RMSE: {hybrid_rmse}")

    # Save metrics
    metrics_data = [
        Row(metric="ALS RMSE", value=float(rmse)),
        Row(metric="Hybrid Model RMSE", value=float(hybrid_rmse)),
    ]

    metrics_df = spark.createDataFrame(metrics_data)
    metrics_df.coalesce(1).write.format("csv").option("header", "true").mode("overwrite") \
              .save(f"{output_path}/metrics.csv")

    return combined_predictions, hybrid_rmse


def find_best_hybrid_weights(
    combined_predictions: DataFrame,
    test: DataFrame,
    weight_values: list = [(0.2, 0.3), (0.3, 0.3), (0.4, 0.3)],
    output_path: str = config.output_path
):
    """
    Try multiple weight combinations for ALS + CF hybrid, pick best RMSE, save best hybrid predictions.

    Returns
    -------
    best_hybrid_df : DataFrame
        Best weighted hybrid predictions
    best_weights : tuple
        (weight_als, weight_cf) that minimizes RMSE
    best_rmse : float
        RMSE for best hybrid model
    """

    test_renamed = test.withColumnRenamed("rating", "true_rating")
    rmse_values = []

    als_prediction_col = "als_prediction"
    cf_prediction_col = "cf_prediction"

    # Evaluate each weight combination
    for weight_als, weight_cf in weight_values:
        hybrid_df = combined_predictions.withColumn(
            "hybrid_prediction",
            F.col(als_prediction_col) * weight_als + F.col(cf_prediction_col) * weight_cf
        )

        evaluation_df = hybrid_df.join(test_renamed, ["userId", "movieId"], "inner") \
                                 .filter(
                                     F.col("true_rating").isNotNull() &
                                     F.col("hybrid_prediction").isNotNull()
                                 )

        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="true_rating", 
            predictionCol="hybrid_prediction"
        )
        rmse_values.append(evaluator.evaluate(evaluation_df))

    best_index = rmse_values.index(min(rmse_values))
    best_weights = weight_values[best_index]

    # Save best hybrid predictions
    best_hybrid_df = combined_predictions.withColumn(
        "hybrid_prediction",
        F.col(als_prediction_col) * best_weights[0] + F.col(cf_prediction_col) * best_weights[1]
    )

    best_hybrid_df.write.mode("overwrite").parquet(f"{s3_path//best_hybrid_model}")

    print(f"Best hybrid model weights: {best_weights}")
    print(f"Best Hybrid Model RMSE: {min(rmse_values)}")

    return best_hybrid_df, best_weights, min(rmse_values)

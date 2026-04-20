from src.models.hybrid_filtering import combine_predictions_and_evaluate, find_best_hybrid_weights
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[2]").appName("Test").getOrCreate()

def test_hybrid():
    # Create dummy ALS predictions
    als_preds = spark.createDataFrame([
        (1, 1, 4.0), (1, 2, 5.0)
    ], ["userId", "movieId", "prediction"])
    
    # Create dummy CF predictions
    cf_preds = spark.createDataFrame([
        (1, 1, 3.5), (1, 2, 4.5)
    ], ["userId", "target_movieId", "prediction"])
    
    test_df = spark.createDataFrame([
        (1, 1, 4.0), (1, 2, 5.0)
    ], ["userId", "movieId", "rating"])
    
    combined, rmse = combine_predictions_and_evaluate(spark, als_preds, cf_preds, test_df, rmse=0)
    assert combined.count() > 0
    assert rmse >= 0

    best_df, best_weights, best_rmse = find_best_hybrid_weights(combined, test_df)
    assert best_df.count() > 0
    assert best_rmse >= 0

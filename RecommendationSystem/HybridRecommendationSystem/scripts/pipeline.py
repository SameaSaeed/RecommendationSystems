import sys
import pandas as pd
from pyspark.sql import SparkSession
from src.utilities.config import config
from data.load_preprocess import load_data, split_raw_data
from model.als_model import train_als_model, evaluate_als_model
from model.item_cf import compute_normalized_ratings, compute_similarity, calculate_item_cf_predictions
from model.hybrid_filtering import combine_predictions_and_evaluate, find_best_hybrid_weights
from mlflow import log_and_register_hybrid_model
from src.serving.serve_endpoint import ModelServing
from src.serving.call_endpoint import call_endpoint

sys.path.append("/Workspace/Users/samiasaeed0006@gmail.com/src")

def main():
    spark = SparkSession.builder.getOrCreate()

    # 1️⃣ Load & split data
    df = load_data(
        spark,
        movie_path=config.MOVIES_CSV,
        rating_path=config.RATINGS_CSV
    )
    train, test = split_raw_data(df)

    # 2️⃣ ALS
    als_model = train_als_model(train)
    rmse, als_predictions_df = evaluate_als_model(als_model, test)

    # 3️⃣ Item-Item CF
    normalized_ratings_df = compute_normalized_ratings(df)
    movie_similarity_df = compute_similarity(normalized_ratings_df)
    user_movie_pairs = test.select("userId").distinct().crossJoin(
        movie_similarity_df.select("movieId1").distinct()
    )
    item_cf_predictions_df = calculate_item_cf_predictions(user_movie_pairs, movie_similarity_df, train)

    # 4️⃣ Hybrid - find best weights and log/register in MLflow
    best_hybrid_df, best_weights, best_rmse = log_and_register_hybrid_model(
        spark,
        als_predictions_df,
        item_cf_predictions_df,
        test,
        model_name="HybridMovieRecommender",
        weight_grid=[(0.4, 0.6), (0.5, 0.5)],
        mlflow_uri="databricks"  # or your local MLflow tracking URI
    )

    print(f"Best hybrid weights: {best_weights}, RMSE: {best_rmse}")

    # 5️⃣ Serve hybrid model in Databricks
    full_model_name = f"{config.catalog_name}.{config.schema_name}.HybridMovieRecommender"
    ms = ModelServing(model_name=full_model_name, endpoint_name=config.endpoint_name)
    ms.deploy_or_update_serving_endpoint(version="latest-model")  # always points to latest

    # 6️⃣ Score some test data via endpoint
    test_sample = best_hybrid_df.select("als_prediction", "cf_prediction").limit(5).toPandas()
    predictions = call_endpoint(test_sample)
    print("Predictions from deployed endpoint:", predictions)

if __name__ == "__main__":
    main()

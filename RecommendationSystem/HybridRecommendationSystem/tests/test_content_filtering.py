from src.models.item_cf import compute_normalized_ratings, compute_similarity, calculate_item_cf_predictions

def test_item_cf_pipeline():
    df = spark.createDataFrame([
        (1, 1, 4.0), (1, 2, 5.0), (2, 1, 3.0)
    ], ["userId", "movieId", "rating"])
    
    norm_df = compute_normalized_ratings(df)
    assert "norm_rating" in norm_df.columns
    
    sim_df = compute_similarity(norm_df)
    assert sim_df.count() > 0
    
    user_movie_pairs = df.select("userId").distinct().crossJoin(sim_df.select("movieId1").distinct())
    preds = calculate_item_cf_predictions(user_movie_pairs, sim_df, df)
    assert preds.count() > 0

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, sqrt, lit, when
from pyspark.sql.window import Window

def compute_normalized_ratings(df: DataFrame) -> DataFrame:
    mean_ratings = df.groupBy("movieId").agg(F.avg("rating").alias("mean_rating"))
    normalized_ratings_df = df.join(mean_ratings, "movieId")
    normalized_ratings_df = normalized_ratings_df.withColumn(
        "norm_rating", col("rating") - col("mean_rating")
    )
    return normalized_ratings_df


def compute_similarity(normalized_ratings_df: DataFrame) -> DataFrame:
    joined_df = normalized_ratings_df.alias("df1").join(normalized_ratings_df.alias("df2"), "userId")
    joined_df = joined_df.filter("df1.movieId < df2.movieId")
    
    joined_df = joined_df.groupBy("df1.movieId", "df2.movieId").agg(
        count(col("df1.movieId")).alias("numPairs"),
        F.sum(col("df1.norm_rating") * col("df2.norm_rating")).alias("sum_xy"),
        F.sum(col("df1.norm_rating") * col("df1.norm_rating")).alias("sum_xx"),
        F.sum(col("df2.norm_rating") * col("df2.norm_rating")).alias("sum_yy")
    )

    result_df = joined_df.withColumn("numerator", col("sum_xy"))
    result_df = result_df.withColumn("denominator", sqrt(col("sum_xx")) * sqrt(col("sum_yy")))
    result_df = result_df.withColumn(
        "similarity",
        when(col("denominator") != 0, col("numerator") / col("denominator")).otherwise(lit(0))
    )

    movie_similarity_df = result_df.select(
        col("df1.movieId").alias("movieId1"),
        col("df2.movieId").alias("movieId2"),
        "similarity"
    )
    return movie_similarity_df


def calculate_item_cf_predictions(user_movie_pairs: DataFrame, movie_similarity_df: DataFrame,
                                  user_ratings_df: DataFrame, N: int = 10) -> DataFrame:
    similarities = movie_similarity_df.alias("sims")
    ratings = user_ratings_df.alias("ratings")
    
    user_item_sims = user_movie_pairs.alias("pairs").join(
        similarities, col("pairs.movieId1") == col("sims.movieId1")
    ).join(
        ratings, (col("ratings.userId") == col("pairs.userId")) & (col("ratings.movieId") == col("sims.movieId2"))
    )

    user_item_sims = user_item_sims.select(
        col("pairs.userId"),
        col("pairs.movieId1").alias("target_movieId"),
        col("sims.movieId2").alias("similar_movieId"),
        col("sims.similarity"),
        col("ratings.rating").alias("similar_movie_rating")
    )

    windowSpec = Window.partitionBy("userId", "target_movieId").orderBy(col("similarity").desc())
    top_n_similar = user_item_sims.withColumn("rank", F.rank().over(windowSpec)).filter(col("rank") <= N)

    weighted_ratings = top_n_similar.groupBy("userId", "target_movieId").agg(
        F.sum(col("similarity") * col("similar_movie_rating")).alias("weighted_sum"),
        F.sum("similarity").alias("similarity_sum")
    )

    predictions = weighted_ratings.withColumn(
        "prediction",
        F.when(col("similarity_sum") != 0, col("weighted_sum") / col("similarity_sum")).otherwise(F.lit(None))
    ).select("userId", "target_movieId", "prediction")

    return predictions

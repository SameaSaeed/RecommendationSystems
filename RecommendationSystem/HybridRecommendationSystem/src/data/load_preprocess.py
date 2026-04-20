from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def load_data(spark: SparkSession, movie_path: str, rating_path: str) -> DataFrame:
    """
    Load movie and rating CSVs, join on movieId, and return full dataframe
    """
    df_movie = spark.read.csv(movie_path, header=True, inferSchema=True)
    df_rating = spark.read.csv(rating_path, header=True, inferSchema=True)
    
    df = df_rating.join(df_movie, on="movieId", how="inner").orderBy("userId")
    return df


def split_raw_data(df: DataFrame, seed: int = 42):
    """
    Split dataframe into train and test sets
    """
    train, test = df.randomSplit([0.8, 0.2], seed=seed)
    return train, test

def user_movie_df(df):
    user_movie_df = (df.groupBy("userId").pivot("title").agg(F.first("rating")))
    return user_movie_df



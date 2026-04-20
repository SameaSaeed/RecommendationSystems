from pyspark.sql import SparkSession
from src.data.load import load_data, split_raw_data

spark = SparkSession.builder.master("local[2]").appName("Test").getOrCreate()

def test_load_data():
    df = load_data(spark, movie_path="tests/sample_movies.csv", rating_path="tests/sample_ratings.csv")
    assert df.count() > 0
    assert "userId" in df.columns
    assert "movieId" in df.columns
    assert "rating" in df.columns

def test_split_raw_data():
    df = load_data(spark, movie_path="tests/sample_movies.csv", rating_path="tests/sample_ratings.csv")
    train, test = split_raw_data(df)
    assert train.count() > 0
    assert test.count() > 0

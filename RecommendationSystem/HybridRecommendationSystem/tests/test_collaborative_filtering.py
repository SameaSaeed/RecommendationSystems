from src.models.als_model import train_als_model, evaluate_als_model

def test_als_training():
    train_data = spark.createDataFrame([
        (1, 1, 4.0), (1, 2, 5.0), (2, 1, 3.0)
    ], ["userId", "movieId", "rating"])
    
    model = train_als_model(train_data)
    assert model is not None

def test_als_evaluation():
    train_data = spark.createDataFrame([
        (1, 1, 4.0), (1, 2, 5.0), (2, 1, 3.0)
    ], ["userId", "movieId", "rating"])
    
    test_data = spark.createDataFrame([
        (2, 2, 4.0)
    ], ["userId", "movieId", "rating"])
    
    model = train_als_model(train_data)
    rmse, predictions = evaluate_als_model(model, test_data)
    assert rmse >= 0
    assert predictions.count() == test_data.count()

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

def train_als_model(train: DataFrame, maxIter=5, regParam=0.01) -> ALS:
    als = ALS(
        maxIter=maxIter,
        regParam=regParam,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train)
    return model


def evaluate_als_model(model: ALS, test: DataFrame) -> tuple:
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse, predictions

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Row
from src.model.hybrid_filtering import combine_predictions_and_evaluate, find_best_hybrid_weights

class HybridRecommendationModel(mlflow.pyfunc.PythonModel):
    def __init__(self, als_weight=0.5, cf_weight=0.5):
        self.als_weight = als_weight
        self.cf_weight = cf_weight

    def predict(self, context, model_input):
        als_predictions = model_input['als_prediction']
        cf_predictions = model_input['cf_prediction']
        return als_predictions * self.als_weight + cf_predictions * self.cf_weight


def log_model(model, model_name: str, X_sample: pd.DataFrame, metrics: dict = None, register=True, mlflow_uri=None):
    """
    Log ML model (ALS or sklearn) to MLflow with optional registration.
    """
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    signature = infer_signature(X_sample, X_sample)
    with mlflow.start_run() as run:
        if metrics:
            mlflow.log_metrics(metrics)

        if register:
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                signature=signature,
                input_example=X_sample,
                registered_model_name=model_name
            )
            version = model_info.registered_model_version
            print(f"Registered model '{model_name}', version {version}")
            return model_info
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                signature=signature,
                input_example=X_sample
            )
            print(f"Model '{model_name}' logged without registration.")
            return None


def log_and_register_hybrid_model(
    spark: SparkSession,
    als_predictions_df: DataFrame,
    cf_predictions_df: DataFrame,
    test_df: DataFrame,
    model_name: str = "HybridMovieRecommender",
    weight_grid: list = [(0.4, 0.6), (0.5, 0.5)],
    mlflow_uri: str = None
):
    """
    Find best hybrid weights, log and register hybrid model in MLflow, and assign 'latest' alias.
    """
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    # 1️⃣ Combine ALS & CF
    rmse = None
    combined_predictions_df, _ = combine_predictions_and_evaluate(
        spark, als_predictions_df, cf_predictions_df, test_df, rmse
    )

    # 2️⃣ Find best weights
    best_hybrid_df, best_weights, best_rmse = find_best_hybrid_weights(
        combined_predictions_df, test_df, weight_values=weight_grid
    )

    # 3️⃣ Prepare sample for signature
    input_example = best_hybrid_df.limit(3).toPandas()

    # 4️⃣ Log and register hybrid model
    hybrid_model = HybridRecommendationModel(als_weight=best_weights[0], cf_weight=best_weights[1])
    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=hybrid_model,
            signature=infer_signature(input_example, input_example),
            input_example=input_example,
            registered_model_name=model_name
        )

    # Fetch the latest version just registered
    latest_version_info = client.get_latest_versions(name=model_name, stages=None)[-1]
    latest_version = latest_version_info.version

    # Set alias 'latest'
    client.set_registered_model_alias(model_name, "latest-model", latest_version)

    print(f"Hybrid model registered with weights {best_weights}, RMSE {best_rmse}, and alias 'latest-model'")

    return best_hybrid_df, best_weights, best_rmse

from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc
import yaml

# ----------------------------
# Load config
# ----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

S3_PARQUET_PATH = config["s3_path"]
MLFLOW_MODEL_URI = config["mlflow_model_uri"]

# ----------------------------
# Load model and predictions
# ----------------------------
hybrid_model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
W_ALS = float(hybrid_model.metadata.get("weight_als", 0.5))
W_CF  = float(hybrid_model.metadata.get("weight_cf", 0.5))
df = pd.read_parquet(S3_PATH, engine="pyarrow")

# ----------------------------
# API
# ----------------------------
app = FastAPI()

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10):
    user_df = (
        df[df["userId"] == user_id]
        .sort_values("hybrid_prediction", ascending=False)
        .head(top_k)
    )
    user_df["hybrid_prediction"] = (
        user_df["als_prediction"] * W_ALS +
        user_df["cf_prediction"] * W_CF
    )
    return user_df.to_dict(orient="records")

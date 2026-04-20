import streamlit as st
import pandas as pd
import mlflow.pyfunc
import yaml

# ----------------------------
# Load config
# ----------------------------
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
S3_PARQUET_PATH = config["s3_path"]
MLFLOW_MODEL_URI = config["mlflow_model_uri"]

# ----------------------------
# Load MLflow model
# ----------------------------
@st.cache_resource
def load_mlflow_model():
    return mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)

hybrid_model = load_mlflow_model()

# ----------------------------
# Get hybrid weights from MLflow model metadata
# ----------------------------
# Make sure you logged weights as params when saving the model
W_ALS = float(hybrid_model.metadata.get("weight_als", 0.5))
W_CF  = float(hybrid_model.metadata.get("weight_cf", 0.5))

st.sidebar.markdown("**Hybrid Weights**")
st.sidebar.markdown(f"- ALS: `{W_ALS}`")
st.sidebar.markdown(f"- CF: `{W_CF}`")

# ----------------------------
# Load predictions from S3 Parquet
# ----------------------------
@st.cache_data
def load_predictions():
    return pd.read_parquet(S3_PARQUET_PATH, engine="pyarrow")

df = load_predictions()

# ----------------------------
# Optional: load movie titles (local)
# ----------------------------
@st.cache_data
def load_movies():
    try:
        return pd.read_csv("movies.csv")  # movieId, title
    except FileNotFoundError:
        return None

movies = load_movies()

# ----------------------------
# Compute hybrid prediction
# ----------------------------
df["hybrid_prediction"] = (
    df["als_prediction"] * W_ALS +
    df["cf_prediction"]  * W_CF
)

# ----------------------------
# Recommendation function
# ----------------------------
def recommend(user_id, k):
    user_df = (
        df[df["userId"] == user_id]
        .sort_values("hybrid_prediction", ascending=False)
        .head(k)
    )

    if movies is not None:
        user_df = user_df.merge(movies, on="movieId", how="left")

    return user_df

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Hybrid Recommendation System")
st.caption("ALS + Item-CF | Weighted Hybrid")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
top_k = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    recs = recommend(user_id, top_k)

    if recs.empty:
        st.warning("No recommendations found for this user.")
    else:
        display_cols = ["movieId", "hybrid_prediction"]
        if "title" in recs.columns:
            display_cols.insert(1, "title")

        st.dataframe(recs[display_cols], use_container_width=True)

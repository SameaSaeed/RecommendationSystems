import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load hybrid model (weights) from pickle
# ----------------------------
@st.cache_resource
def load_hybrid_model():
    with open("hybrid_model.pkl", "rb") as f:
        return pickle.load(f)

hybrid = load_hybrid_model()
W_ALS = hybrid["weight_als"]
W_CF  = hybrid["weight_cf"]

st.sidebar.markdown(f"**Hybrid Weights:** ALS={W_ALS}, CF={W_CF}")

# ----------------------------
# Load predictions (Parquet exported from Databricks)
# Columns: userId, movieId, als_prediction, cf_prediction
# ----------------------------
@st.cache_data
def load_predictions():
    return pd.read_parquet("best_hybrid_model.parquet")

df = load_predictions()

# ----------------------------
# Optional: load movie titles
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
def recommend(user_id, k=10):
    user_df = df[df["userId"] == user_id] \
        .sort_values("hybrid_prediction", ascending=False) \
        .head(k)

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

        st.dataframe(
            recs[display_cols],
            use_container_width=True
        )

# End-to-End Hybrid Recommendation System

## Introduction

The End-to-End Hybrid Recommendation System leverages Databricks to build a scalable, flexible, and production-ready recommendation engine. This repository demonstrates how to preprocess data, train collaborative and content-based models, blend predictions, and deploy recommendations efficiently within the Databricks ecosystem. It is designed for online retail, e-commerce, or any domain requiring personalized item suggestions.

## Features

- End-to-end data pipeline: ingestion, preprocessing, feature engineering, model training, and prediction.
- Hybrid approach: blends collaborative filtering (ALS) and content-based filtering for improved accuracy.
- Scalable architecture using Apache Spark and Databricks.
- Model evaluation and hyperparameter tuning.
- Model deployment and batch/real-time inference support.
- Modular notebooks and scripts for extensibility.

## Requirements

- Databricks workspace (Community, Enterprise, or Azure/AWS Databricks)
- Databricks Runtime 8.1+ (with ML support recommended)
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - pyspark
  - matplotlib (for visualization)
- Data: User-item interaction logs and item metadata (CSV, Parquet, or Delta format supported)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SameaSaeed/End-to-End_HybridRecommendationSystem_DataBricks.git
   cd End-to-End_HybridRecommendationSystem_DataBricks
   ```

2. **Import notebooks into Databricks:**
   - Go to your Databricks workspace.
   - Use the UI to import each notebook from the repository (`/notebooks` folder).

3. **Install required Python libraries:**
   - In a Databricks notebook cell or a cluster init script, run:
     ```python
     %pip install pandas numpy scikit-learn matplotlib
     ```

4. **Upload your data:**
   - Place your user-item interactions and item metadata files in Databricks File System (DBFS) or accessible cloud storage.

## Usage

1. **Data Ingestion and Preprocessing:**
   - Start with the `01_data_ingestion_preprocessing` notebook.
   - Configure the data paths and run all cells to clean and prepare your datasets.

2. **Feature Engineering:**
   - Open `02_feature_engineering`.
   - Generate user/item features and encode categorical variables.

3. **Collaborative Filtering (ALS):**
   - Use `03_collaborative_filtering_ALS` to train an ALS model.
   - Evaluate the model and save predictions.

4. **Content-Based Filtering:**
   - Proceed with `04_content_based_filtering`.
   - Train and evaluate a content-based recommender using item features.

5. **Hybrid Model Blending:**
   - The `05_hybrid_blending` notebook combines predictions from both models.
   - Tune blending ratios and evaluate hybrid performance.

6. **Model Deployment:**
   - Use `06_model_deployment` to save models and generate recommendations for new users.

7. **Model Inferencing:**

<img width="1254" height="731" alt="hybrid_prediction" src="https://github.com/user-attachments/assets/f0db5ad5-5f26-427e-90e4-2dedebf7bd6a" />




For questions, open an issue or contact the repository maintainer via GitHub.

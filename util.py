from math import sqrt

from pyspark.ml.connect.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
import streamlit as st
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, avg, corr
import pandas as pd

spark = SparkSession.builder.appName("AirlineReviews").getOrCreate()


def get_data():
    ratings_df = spark.read.csv(
        "data/airlines_reviews.csv",
        sep=",",  # Delimiter
        inferSchema=True,  # Infer data types
        header=True,  # First row as header
        quote='"',  # Handle fields enclosed in double quotes
        escape='"',  # Escape character for quotes within fields
        multiLine=True  # Enable parsing of multi-line fields
    )
    ratings_df = ratings_df.toDF(
        "Title", "Name", "Review Date", "Airline", "Verified", "Reviews", "Type of Traveller", "Month Flown",
        "Route", "Class", "Seat Comfort", "Staff Service", "Food & Beverages", "Inflight Entertainment",
        "Value For Money", "Overall Rating", "Recommended"
    )
    return ratings_df


def get_aggregations(df):
    return df.groupBy("Airline").agg(
        avg("Overall Rating").alias("Avg_Rating"),
        avg("Seat Comfort").alias("Avg_Seat_Comfort"),
        avg("Staff Service").alias("Avg_Staff_Service"),
        avg("Food & Beverages").alias("Avg_Food_Beverages"),
        avg("Inflight Entertainment").alias("Avg_Inflight_Entertainment"),
        avg("Value For Money").alias("Avg_Value_For_Money")
    ).orderBy("Avg_Rating", ascending=False)


def get_correlations(df):
    correlation_columns = [
        "Overall Rating", "Seat Comfort", "Staff Service", "Food & Beverages",
        "Inflight Entertainment", "Value For Money"
    ]

    # Ensure the columns are of numeric type
    for col_name in correlation_columns:
        df = df.withColumn(col_name, df[col_name].cast("float"))

    correlations = []
    for col1 in correlation_columns:
        row = []
        for col2 in correlation_columns:
            if col1 == col2:
                row.append(1.0)  # Correlation of a column with itself is 1
            else:
                corr_value = df.select(corr(col1, col2)).first()[0]
                row.append(corr_value)
        correlations.append(row)

    # Creating a DataFrame for the correlation matrix
    correlation_df = pd.DataFrame(correlations, columns=correlation_columns, index=correlation_columns)
    return correlation_df


def train_sentiment_analysis_model(df):
    # Tokenize the reviews into words
    tokenizer = Tokenizer(inputCol="Reviews", outputCol="words")

    # Remove stop words
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Convert words into term frequency (TF) using HashingTF
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)

    # Apply IDF (Inverse Document Frequency) to weigh the terms
    idf = IDF(inputCol="raw_features", outputCol="features")

    # If > 5 positif else negatif
    df_with_labels = df.withColumn("label", (col("Recommended") == "yes").cast("int"))

    # Split the data into training and test sets (80% train, 20% test)
    train_data, test_data = df_with_labels.randomSplit([0.8, 0.2], seed=1234)

    # Create a Logistic Regression classifier
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    # Create the full pipeline with the classifier
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])

    # Train the model
    model = pipeline.fit(train_data)

    # Evaluate the model on the test data
    predictions = model.transform(test_data)

    # Evaluate the performance of the model
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    st.metric(
        label="NLP Model Accuracy",
        value="{:.2f}%".format(evaluator.evaluate(predictions) * 100)
    )

    df_with_predictions = model.transform(df_with_labels)
    df_with_predictions = df_with_predictions.withColumn(
        "Predicted_Sentiment",
        when(col("prediction") == 1, "Positif").otherwise("Négatif")
    )

    return df_with_predictions


def get_prediction_model(df_with_predictions):
    df_with_sentiment_score = df_with_predictions.withColumn(
        "Sentiment_Score",
        when(col("Predicted_Sentiment") == "positive", 1).otherwise(0)
    )

    # relevant columns for prediction
    features = ["Seat Comfort", "Staff Service", "Food & Beverages",
                "Inflight Entertainment", "Value For Money",
                "Sentiment_Score"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    # Prepare data for training
    if "features" in df_with_sentiment_score.columns:
        df_with_sentiment_score = df_with_sentiment_score.withColumnRenamed("features", "old_features")
    if "prediction" in df_with_sentiment_score.columns:
        df_with_sentiment_score = df_with_sentiment_score.withColumnRenamed("prediction", "old_prediction")
    df_features = assembler.transform(df_with_sentiment_score)

    # Split data into training and testing sets (80% train, 20% test)
    train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)

    # Train a Random Forest Regressor model to predict Overall Rating
    rf = RandomForestRegressor(labelCol="Overall Rating", featuresCol="features", numTrees=50)
    model = rf.fit(train_data)

    # Make predictions on the test data
    predictions = model.transform(test_data)

    # Evaluate the model using RegressionEvaluator
    squared_diff = predictions.withColumn("squared_diff", (col("Overall Rating") - col("prediction"))**2)
    mean_squared_diff = squared_diff.select(mean("squared_diff")).collect()[0][0]
    rmse = sqrt(mean_squared_diff)

    # Show RMSE of the model
    st.metric(
        "RMSE du modèle Random Forest:",
        "{:.2f}".format(rmse)
    )

    return predictions
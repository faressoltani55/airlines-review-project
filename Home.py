import streamlit as st
from pyspark.sql.functions import count, col, when, mean
import matplotlib.pyplot as plt
from util import get_data, get_aggregations, get_correlations, train_sentiment_analysis_model, get_prediction_model
import seaborn as sns

st.set_page_config(layout="wide")

st.header(
    "Prédiction de la satisfaction client dans l’aviation",
)
st.divider()

tabs = st.tabs(["Base de Données", "Analyses", "Outil de Prédiction"])
data = get_data()

# Filter par Airline et Type de Voyageur
st.sidebar.header("Filters")
airlines = data.select("Airline").distinct().toPandas()["Airline"].to_list()
selected_airline = st.sidebar.selectbox("Select Airline", ["All"] + airlines)
traveller_types = data.select("Type of Traveller").distinct().toPandas()["Type of Traveller"].to_list()
selected_traveller_type = st.sidebar.selectbox("Select Traveller Type", ["All"] + traveller_types)

filtered_df = data
if selected_airline != "All":
    filtered_df = data.filter(col("Airline") == selected_airline)
if selected_traveller_type != "All":
    filtered_df = data.filter(col("Type of Traveller") == selected_traveller_type)

with tabs[0]:
    tabs[0].header("Base de données:")
    tabs[0].dataframe(
        data=filtered_df,
        use_container_width=True
    )
    tabs[0].header("Analyse de Base:")
    basic_analysis_columns = tabs[0].columns(2)

    total_rows = filtered_df.count()
    total_columns = len(filtered_df.columns)
    missing_values = filtered_df.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in filtered_df.columns]
    ).toPandas()

    basic_analysis_columns[0].metric(
        label="Nombre de lignes",
        value=total_rows
    )
    basic_analysis_columns[1].metric(
        label="Nombre de colonnes",
        value=total_columns
    )
    basic_analysis_columns[0].metric(
        label="Nombre de Airlines",
        value=filtered_df.select("Airline").distinct().count()
    )
    basic_analysis_columns[1].metric(
        label="Nombre d'avis vérifiés",
        value=filtered_df.filter(col("Verified") == True).count()
    )
    tabs[0].subheader("Valeurs manquantes")
    tabs[0].dataframe(
        missing_values,
        use_container_width=True
    )

with tabs[1]:
    tabs[1].header("Analyse Globale des KPIs")
    # Average Overall Rating
    avg_rating = filtered_df.select(mean(col("Overall Rating").cast("float"))).first()[0]

    # Percentage Recommended
    recommended_percentage = (
            filtered_df.filter(col("Recommended") == "yes").count() / total_rows * 100
    )

    # Top 5 Airlines by Overall Rating
    top_airlines = (
        filtered_df.groupBy("Airline")
        .agg(
            mean(col("Overall Rating").cast("float")).alias("AvgRating"),
            count("*").alias("ReviewCount")
        )
        .orderBy(col("AvgRating").desc())
        .limit(5)
        .toPandas()
    )

    kpis_columns = tabs[1].columns(2)
    kpis_columns[0].metric(
        "Score Général Moyen",
        f"{avg_rating:.2f}"
    )
    kpis_columns[1].metric(
        "Percentage de Recommendation",
        f"{recommended_percentage:.2f}%"
    )
    tabs[1].subheader("Top 3 Companies Aériennes par Rating Moyen:")
    tabs[1].dataframe(
        top_airlines,
        use_container_width=True
    )

    # Visualisation de la distribution
    analysis_viz_columns = tabs[1].columns(2)
    ratings_count = (
        filtered_df.groupBy("Overall Rating")
        .count()
        .orderBy(col("Overall Rating").cast("float"))
        .toPandas()
    )

    fig, ax = plt.subplots()
    ax.bar(ratings_count["Overall Rating"], ratings_count["count"], color='skyblue')
    ax.set_title("Distribution of Overall Rating")
    ax.set_xlabel("Overall Rating")
    ax.set_ylabel("Count")
    analysis_viz_columns[0].pyplot(
        fig
    )

    # Pie Chart of Recommended
    recommended_data = (
        filtered_df.groupBy("Recommended")
        .count()
        .toPandas()
    )

    fig, ax = plt.subplots()
    ax.pie(
        recommended_data["count"],
        labels=recommended_data["Recommended"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["red", "green"],
    )
    ax.set_title("Recommended Percentage")
    analysis_viz_columns[1].pyplot(fig)

    tabs[1].header("Analyse Approfondie")
    tabs[1].subheader("Rating Moyen Par Airline :")
    tabs[1].dataframe(
        get_aggregations(filtered_df),
        use_container_width=True
    )

    tabs[1].subheader("Analyse de Corrélation: ")
    correlation_matrix = get_correlations(data)
    st.title("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True
    )
    tabs[1].pyplot(fig)

with tabs[2]:
    # Sentiment Analysis:
    df_with_predictions = train_sentiment_analysis_model(filtered_df)

    tabs[2].header("Analyse des Sentiments des Reviews: ")
    tabs[2].dataframe(
        df_with_predictions.select("Reviews", "Predicted_Sentiment", "Overall Rating"),
        use_container_width=True
    )

    import matplotlib.pyplot as plt

    # Count the number of positive and negative reviews
    sentiment_counts = df_with_predictions.groupBy("Predicted_Sentiment").count().toPandas()

    # Plot the sentiment distribution
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts["count"],
        labels=sentiment_counts["Predicted_Sentiment"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["red", "green"],
    )
    ax.set_title("Predicted Sentiment Distribution")
    tabs[2].pyplot(fig)

    # Analyse Prédictive
    prediction_result_df = get_prediction_model(df_with_predictions).toPandas()
    fig, ax = plt.subplots()
    ax.scatter(prediction_result_df["Overall Rating"], prediction_result_df["prediction"], alpha=0.5)
    ax.plot([1, 5], [1, 5], color='red', linestyle='--')
    ax.set_xlabel("Actual Overall Rating")
    ax.set_ylabel("Predicted Overall Rating")
    ax.set_title("Actual vs Predicted Overall Rating")
    st.pyplot(fig)


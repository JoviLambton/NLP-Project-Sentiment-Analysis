import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


model = joblib.load('clf_nb.pkl')
vectorizer = joblib.load('tfidf.pkl')


def process_file(df):
    text_df = df["Review"]
    text_vectorized = vectorizer.transform(text_df)
    df["Sentiment"] = model.predict(text_vectorized)
    df["Sentiment"] = df["Sentiment"].apply(lambda x: "Negative" if x == 0 else "Positive")
    # st.dataframe(df)
    return df

def main():
    st.title("Sentiment Analysis Report")

    # File Upload
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])
    
    # Multi-select for Products
    selected_products = st.multiselect("Select Products", ["Timestamp", "Source", "Product", "Topic"])

    if uploaded_file is not None:
        # st.text("Selected: "+ str(selected_products))

        # Read file
        if uploaded_file.type == 'text/csv':
            # Text file
            df = pd.read_csv(uploaded_file, encoding="latin-1")

        if st.button("Process Text"):
            df = process_file(df)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Streamlit App
            st.title("Product Sentiment Analysis Report")

            # Sentiment Distribution
            st.header("Sentiment Distribution")
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            st.table(sentiment_counts)

            # Time Series Analysis
            st.header("Time Series Analysis")
            if 'Timestamp' in df.columns:
                time_df = df[['Timestamp', 'Sentiment', 'Source']]
                time_df = df.groupby(['Timestamp', 'Sentiment']).count().fillna(0)
                time_df = time_df.reset_index()
                time_df = pd.pivot_table(time_df, values='Source', index=['Timestamp'], columns='Sentiment', aggfunc="sum").fillna(0)
                time_df = time_df.reset_index()
                st.dataframe(time_df)
                chart_data = pd.DataFrame()
                chart_data['Timestamp'] = time_df['Timestamp']
                chart_data['Negative'] = time_df['Negative']
                # chart_data['Neutral'] = time_df['Neutral']
                chart_data['Positive'] = time_df['Positive']
                st.line_chart(chart_data.set_index('Timestamp'))


            st.header("Source Analysis")
            source_counts = df['Source'].value_counts()
            st.bar_chart(source_counts)
            st.table(source_counts)

            # # Topic Analysis
            # st.header("Topic Analysis")
            # topic_counts = df['Review'].str.lower().value_counts()
            # st.bar_chart(topic_counts)
            # st.table(topic_counts)

            # Comparison Across Products/Features
            st.header("Comparison Across Products/Features")
            if 'Product' in df.columns:
                # Filter data based on selected products
                product_sentiment_counts = df[['Product', 'Sentiment', 'Source']].groupby(['Product', 'Sentiment']).count().fillna(0)
                product_sentiment_counts = product_sentiment_counts.rename(columns={"Source": "Count"})
                product_sentiment_counts = product_sentiment_counts.reset_index()
                
                bar_chart = alt.Chart(product_sentiment_counts).mark_bar().encode(
                        x="Product",
                        y="Count",
                        color="Sentiment",
                    )
                st.altair_chart(bar_chart, use_container_width=True)

                st.table(product_sentiment_counts)


if __name__ == "__main__":
    main()

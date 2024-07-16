import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download Vader lexicon if not already downloaded
nltk.download('vader_lexicon')

# Function to perform sentiment analysis and assign sentiment labels
def perform_sentiment_analysis(data):
    sid = SentimentIntensityAnalyzer()
    data['sentiment'] = [sid.polarity_scores(data.text.iloc[i])['compound'] for i in range(len(data))]
    data['feel'] = data.sentiment.apply(sentiment_class)
    return data

# Function to classify sentiment
def sentiment_class(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to plot pie chart
def plot_pie_chart(data):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10,10))
    ax.pie(list(data.feel.value_counts()), 
           labels=data.feel.value_counts().index, 
           autopct='%1.1f%%',
           wedgeprops={'linewidth': 7, 'edgecolor': 'whitesmoke'})
    circle = plt.Circle((0,0), 0.3, color='whitesmoke')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    ax.set_title('Count of Tweets')
    ax.axis('equal')
    st.pyplot(fig)

# Function to plot histogram
def plot_histogram(data):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(data['sentiment'], bins=5)
    plt.title("Sentiment")
    ax.set_xticks([-1,0,1])
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    plt.xlabel("Sentiment")
    plt.ylabel("# of Tweets")
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Stock Text Sentiment Analysis")
    st.sidebar.title("Settings")

    # File uploader for dataset
    uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Read dataset
            data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
            st.dataframe(data.head())

            # Perform sentiment analysis
            analyzed_data = perform_sentiment_analysis(data)

            # Display sentiment counts
            st.write("### Sentiment Counts")
            st.write(analyzed_data.feel.value_counts())

            # Plot pie chart
            st.write("### Pie Chart of Sentiment Distribution")
            plot_pie_chart(analyzed_data)

            # Plot histogram
            st.write("### Histogram of Sentiment Scores")
            plot_histogram(analyzed_data)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import requests

# Function to retrieve quarterly high and low prices and classify reports
def get_quarterly_report(ticker):
    stock = yf.Ticker(ticker)

    # Get historical data for the last 3-4 years
    hist = stock.history(period="4y")

    # Resample to quarterly intervals and calculate high and low prices
    quarterly_data = hist.resample('Q').agg({'High': 'max', 'Low': 'min'})

    # Use linear regression to predict future prices for the next quarter
    X = pd.DataFrame({'Quarter': range(len(quarterly_data))}).values.reshape(-1, 1)
    y_high = quarterly_data['High'].values
    y_low = quarterly_data['Low'].values

    model_high = LinearRegression().fit(X, y_high)
    model_low = LinearRegression().fit(X, y_low)

    future_quarter = len(quarterly_data)
    next_quarter = future_quarter + 1

    future_price_high = model_high.predict([[next_quarter]])[0]
    future_price_low = model_low.predict([[next_quarter]])[0]

    # Classify the next quarter's report based on the predicted future prices
    decision = 'Hold'
    if (future_price_high >= 1.05 * quarterly_data['High'].iloc[-1]) and (future_price_low >= 1.05 * quarterly_data['Low'].iloc[-1]):
        decision = 'Buy'
    elif (future_price_high <= 0.95 * quarterly_data['High'].iloc[-1]) and (future_price_low <= 0.95 * quarterly_data['Low'].iloc[-1]):
        decision = 'Sell'

    return quarterly_data, future_price_high, future_price_low, decision

# Streamlit app
st.title('Stock Price Prediction for Next Quarter')
ticker = st.text_input('Enter Ticker Symbol (e.g., AAPL for Apple Inc.)')

if st.button('Predict'):
    if not ticker:
        st.error('Please enter a valid ticker symbol.')
    else:
        try:
            quarterly_report, future_price_high, future_price_low, decision = get_quarterly_report(ticker.upper())
            st.header('Quarterly High and Low Prices:')
            st.write(quarterly_report)
            st.header('Predicted Prices for Next Quarter:')
            st.write(f'Predicted High: {future_price_high:.2f}')
            st.write(f'Predicted Low: {future_price_low:.2f}')
            st.success(f'Final Decision for Next Quarter: {decision}')

            # API endpoint for the selected ticker fundamentals
            url = f'https://eodhd.com/api/fundamentals/{ticker}?api_token=demo&fmt=json'
            data = requests.get(url).json()

            # Extracting required information
            if 'Message' in data:
                st.error("Invalid Ticker Symbol. Please enter a valid ticker symbol.")
            else:
                trailing_pe = data['Valuation']['TrailingPE']
                forward_pe = data['Valuation']['ForwardPE']
                eps_current_year = data['Highlights']['EarningsShare']
                eps_next_year = data['Highlights']['EPSEstimateNextYear']
                price_to_book = data['Valuation']['PriceBookMRQ']
                overall_rating = data['AnalystRatings']['Rating']
                target_price = data['AnalystRatings']['TargetPrice']
                strong_buy = data['AnalystRatings']['StrongBuy']
                buy = data['AnalystRatings']['Buy']
                hold = data['AnalystRatings']['Hold']
                sell = data['AnalystRatings']['Sell']
                strong_sell = data['AnalystRatings']['StrongSell']

                # Displaying the extracted information
                st.subheader("Stock Fundamentals")
                st.write(f"*Ticker Symbol:* {ticker}")
                st.write(f"*Trailing P/E Ratio:* {trailing_pe}")
                st.write(f"*Forward P/E Ratio:* {forward_pe}")
                st.write(f"*Earnings Per Share (EPS) Current Year:* {eps_current_year}")
                st.write(f"*Earnings Per Share (EPS) Next Year:* {eps_next_year}")
                st.write(f"*Price-to-Book Ratio:* {price_to_book}")
                st.write(f"*Overall Rating:* {overall_rating}")
                st.write(f"*Target Price:* {target_price}")
                st.write(f"*Strong Buy:* {strong_buy}")
                st.write(f"*Buy:* {buy}")
                st.write(f"*Hold:* {hold}")
                st.write(f"*Sell:* {sell}")
                st.write(f"*Strong Sell:* {strong_sell}")
        except requests.exceptions.RequestException as e:
            st.error("Failed to fetch fundamentals data. Please try again later.")
        except KeyError as e:
            st.error("Failed to extract information from API response. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
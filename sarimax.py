import streamlit as st 
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Function to perform Augmented Dickey-Fuller test for stationarity
def adfuller_test(series):
    result = adfuller(series)
    return result[1]

# Title
app_name='Stock Market Forecast'
st.title(app_name)
st.subheader('Forecast the stock market price of the selected company')
# Add an image using online source
st.image("https://resize.indiatvnews.com/en/resize/newbucket/400_-/2021/09/stockmarket-1631258083.jpg")

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

start_date=st.sidebar.date_input('Start Date',date(2023,1,1))
end_date=st.sidebar.date_input('End Date',date(2024,3,31))
# Add ticker symbol list
ticker = st.sidebar.text_input('Enter the ticker symbol (e.g., AAPL)', 'AAPL')


# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)

# Convert the index to DatetimeIndex
data.index = pd.to_datetime(data.index)

# Reset the index and set the 'Date' column
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)

st.write('Data from ', start_date, ' to ', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
fig = px.line(data, x='Date', y='Close', title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to select column from data
column=st.selectbox('Select the column to be used for forecasting',data.columns[1:])

# Setting the data
data=data[['Date',column]]
st.write("Selected Data")
st.write(data)

# ADF test check stationarity
st.header('Is data stationary?')
st.write('**Note:** If p-value is less than 0.05, then data is stationary')
st.write(adfuller_test(data[column]))
st.write(adfuller_test(data[column]) < 0.05)

# Decompose the data
st.header('Decomposition of the data')
st.write('**Note:** The data is decomposed into trend, seasonality and residual')
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

# Make the same plot in plotly
st.write("Plotting the decomposition using plotly")
st.plotly_chart(px.line(x=data.index,y=decomposition.trend, title='Trend',width=1000,height=400,labels={'x': 'Date','y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data.index,y=decomposition.seasonal, title='Seasonality',width=1000,height=400,labels={'x': 'Date','y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data.index,y=decomposition.resid, title='Residuals',width=1000,height=400,labels={'x': 'Date','y': 'Price'}).update_traces(line_color='Red',line_dash='dot'))

# User input for three parameters of the model and seasonal order
p=st.slider('Select the value of p',0,5,2)
d=st.slider('Select the value of d',0,5,1)
q=st.slider('Select the value of q',0,5,2)
seasonal_order=st.number_input('Select the value of seasonal p',0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit(disp=-1)

# Print the model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")

# Predict the future values
st.write("<p style='color:green; font-size:50px; font-weight:bold;'>Forecasting the data</p>",unsafe_allow_html=True)
forecast_period=st.number_input('Select the number of days to forecast',1,365,10)
predictions=model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions=predictions.predicted_mean

# Add index to the predictions
predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index,True)
predictions.reset_index(drop=True,inplace=True)
st.write("##Predictions",predictions)
st.write("##Actual Data",data)
st.write("---")

# Plot the data
fig=go.Figure()
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode='lines',name='Predicted',line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Price',width=1000,height=400)
st.plotly_chart(fig)

# Add buttons to show and hide separate plots
show_plots=False
if st.button('Show separate plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title='Actual',width=1000,height=400,labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title='Predicted',width=1000,height=400,labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))
        show_plots=True
    else:
        show_plots=False
        
        
# Calculate the accuracy metrics

def calculate_accuracy(actual, predicted):
    # Ensure that actual and predicted arrays have the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


# Get actual values for the forecast period
actual_values = yf.download(ticker, start=end_date, end=end_date+timedelta(days=forecast_period-1))

# Calculate the accuracy metrics
mae, mse, rmse = calculate_accuracy(actual_values['Close'].values, predictions['predicted_mean'].values)

# Display the accuracy metrics
# Display the accuracy metrics
st.header('Prediction Accuracy')
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}") 
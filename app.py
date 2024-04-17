import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from datetime import date
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import path

model = load_model('/Users/vaishnavimhaske/Documents/Jupyter Notebooks/Stock Prediction Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'AAPL')
start = '2021-01-01'
end = date.today().strftime("%Y-%m-%d")

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Averages 50 Days')
moving_averges_50days =  data.Close.rolling(50).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=moving_averges_50days, mode='lines', name='50-Day Moving Average', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))

fig.update_layout(title='Moving Averages and Closing Prices',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title='Legend')
st.plotly_chart(fig)

st.subheader('Price vs Moving Averages 50 Days vs Moving Averages 100 days')
moving_averges_100days =  data.Close.rolling(100).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=moving_averges_50days, mode='lines', name='50-Day Moving Average', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data.index, y=moving_averges_100days, mode='lines', name='100-Day Moving Average', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))

fig.update_layout(title='Moving Averages and Closing Prices',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title='Legend')
st.plotly_chart(fig)

st.subheader('Price vs Moving Averages 100 Days vs Moving Averages 200 days')
moving_averges_200days =  data.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=moving_averges_100days, mode='lines', name='100-Day Moving Average', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data.index, y=moving_averges_200days, mode='lines', name='200-Day Moving Average', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))

fig.update_layout(title='Moving Averages and Closing Prices',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title='Legend')
st.plotly_chart(fig)

x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

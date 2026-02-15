import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------
# Page config (CENTERED layout)
# ---------------------------------

st.set_page_config(
    page_title="Stock Price Predictor App",
    layout="centered"
)

# ---------------------------------
# Title
# ---------------------------------

st.title("Stock Price Predictor App")

# ---------------------------------
# Stock input
# ---------------------------------

stock = st.text_input("Enter the Stock ID", "GOOG")

# ---------------------------------
# Load data
# ---------------------------------

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

data = yf.download(stock, start, end)

if data.empty:
    st.error("Invalid stock symbol")
    st.stop()

# ---------------------------------
# Show stock data
# ---------------------------------

st.subheader("Stock Data")
st.write(data)

# ---------------------------------
# Moving averages
# ---------------------------------

data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()
data['MA250'] = data['Close'].rolling(250).mean()

# ---------------------------------
# Plot function
# ---------------------------------

def plot_graph(columns, title):

    fig = plt.figure(figsize=(12,5))

    for column in columns:
        plt.plot(data[column], label=column)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    st.pyplot(fig)

# ---------------------------------
# MA graphs
# ---------------------------------

st.subheader("Original Close Price and MA for 250 days")
plot_graph(['Close', 'MA250'], "Close Price vs MA250")

st.subheader("Original Close Price and MA for 200 days")
plot_graph(['Close', 'MA200'], "Close Price vs MA200")

st.subheader("Original Close Price and MA for 100 days")
plot_graph(['Close', 'MA100'], "Close Price vs MA100")

st.subheader("Original Close Price and MA for 100 days and MA for 250 days")
plot_graph(['Close', 'MA100', 'MA250'], "Close vs MA100 vs MA250")

# ---------------------------------
# Load trained model
# ---------------------------------

model = load_model("Latest_stock_price_model_v2.keras")

# ---------------------------------
# Prepare prediction data
# ---------------------------------

features = data[['Open','High','Low','Close','Volume']]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

split = int(len(scaled_data) * 0.7)

test_data = scaled_data[split:]

X_test = []
y_test = []

for i in range(100, len(test_data)):
    
    X_test.append(test_data[i-100:i])
    
    y_test.append(test_data[i,3])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ---------------------------------
# Predict
# ---------------------------------

predictions = model.predict(X_test)

# ---------------------------------
# Inverse transform
# ---------------------------------

dummy_pred = np.zeros((len(predictions),5))
dummy_pred[:,3] = predictions.reshape(-1)

predicted_prices = scaler.inverse_transform(dummy_pred)[:,3]

dummy_actual = np.zeros((len(y_test),5))
dummy_actual[:,3] = y_test.reshape(-1)

actual_prices = scaler.inverse_transform(dummy_actual)[:,3]

# ---------------------------------
# Accuracy metrics
# ---------------------------------

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
accuracy = 100 - (mae / np.mean(actual_prices)) * 100

st.subheader("Model Accuracy")

st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"Accuracy: {accuracy:.2f}%")

# ---------------------------------
# Results table
# ---------------------------------

index = data.index[split+100:]

results = pd.DataFrame({

    "Original Price": actual_prices,
    "Predicted Price": predicted_prices

}, index=index)

st.subheader("Original values vs Predicted values")

st.write(results)

# ---------------------------------
# Prediction graph
# ---------------------------------

st.subheader("Original Close Price vs Predicted Close Price")

fig2 = plt.figure(figsize=(12,5))

plt.plot(data['Close'][:split+100], label="Training Data")

plt.plot(results['Original Price'], label="Actual Price")

plt.plot(results['Predicted Price'], label="Predicted Price")

plt.xlabel("Date")
plt.ylabel("Stock Price")

plt.legend()

st.pyplot(fig2)


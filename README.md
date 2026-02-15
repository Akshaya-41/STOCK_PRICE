# Stock Price Predictor

Live App: https://stock-price-predictor-akshaya.streamlit.app/
This project predicts stock prices using Deep Learning (LSTM) and Streamlit.Users can enter any stock symbol and view predictions, moving averages, and model accuracy.

## Features

- Predict stock prices using trained LSTM model
- Interactive Streamlit web app
- Moving averages visualization (MA100, MA200, MA250)
- Original vs Predicted price comparison
- Accuracy metrics (RMSE, MAE, Accuracy %)

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pandas, NumPy
- Matplotlib
- yFinance

## Project Structure
```sh
STOCK_PRICE/
│
├── web_stock_price_predictor.py
├── stockPrice.ipynb
├── Latest_stock_price_model_v2.keras
├── requirements.txt
└── README.md
```


## How to Run
### Install dependencies:
```sh
pip install -r requirements.txt
```

### Run app:
```sh
streamlit run web_stock_price_predictor.py
```


## Model

- LSTM Deep Learning model
- Uses Open, High, Low, Close, Volume features
- Accuracy: ~95%

## Author

Akshaya Penumathsa



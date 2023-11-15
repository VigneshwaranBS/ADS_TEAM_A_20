import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ccxt
from datetime import datetime

# Initialize the exchange (e.g., Binance)
exchange = ccxt.binance()

st.title('Stock and Crypto Price Prediction App')

# Define a function to fetch stock data
def get_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Define a function to fetch cryptocurrency data
def get_crypto_data(symbol, start, end):
    # Convert the start and end dates to timestamps in milliseconds
    start_timestamp = pd.Timestamp(start).timestamp() * 1000
    end_timestamp = pd.Timestamp(end).timestamp() * 1000

    # Fetch the cryptocurrency data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=int(start_timestamp), limit=500, params={'endTime': int(end_timestamp)})

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Define a function to retrieve news articles related to a symbol
def get_symbol_news(symbol):
    newsapi = NewsApiClient(api_key='ef7280046c0d433bbf74a27166e796b5')  # Replace with your News API key
    news = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page=1, page_size=5)
    return news

# Define a function for sentiment analysis using VADER
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

# Get a list of available stock symbols
available_stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'INFY', 'WMT', 'IBM', 'MU', 'BA', 'AXP']

# Get a list of available cryptocurrency symbols
available_crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'DOGE/USDT']

# Add more stock symbols
additional_stock_symbols = ['TSLA', 'GOOG', 'FB', 'NFLX', 'INTC']

# Add more cryptocurrency symbols
additional_crypto_symbols = ['BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'MATIC/USDT', 'DOT/USDT']

# Combine the additional symbols with the existing symbols
available_stock_symbols += additional_stock_symbols
available_crypto_symbols += additional_crypto_symbols

# Sidebar - Input parameters
st.sidebar.header('Input Parameters')

# Create a multi-select dropdown for selecting stock symbols
selected_stock_symbols = st.sidebar.multiselect('Select Stock Symbols', available_stock_symbols)

# Create a multi-select dropdown for selecting cryptocurrency symbols
selected_crypto_symbols = st.sidebar.multiselect('Select Crypto Symbols', available_crypto_symbols)

# Create date pickers for start and end dates
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2018-01-01'))
end_date = st.sidebar.date_input('End Date', datetime.today())

# Create input fields for investment details
st.sidebar.subheader('Investment Details')
investment_type = st.sidebar.radio('Select Stock or Crypto:', ['Stock', 'Crypto'])
investment_symbol = ''
if investment_type == 'Crypto':
    investment_symbol = st.sidebar.selectbox('Select Crypto Symbol:', available_crypto_symbols)
else:
    investment_symbol = st.sidebar.selectbox('Select Stocks Symbol:', available_stock_symbols)
investment_amount = st.sidebar.number_input('Amount Invested (in dollars):', min_value=0.01)
investment_date = st.sidebar.date_input('Date of Investment', datetime.now().date())
end_date_investment = min(investment_date, datetime.now().date())

# Calculate investment results
if investment_symbol and investment_amount and investment_date:
    st.subheader('Investment Results')
    end_date_investment = min(investment_date, datetime.now().date())
    
    if investment_type == 'Stock':
        # Calculate for stocks
        investment_data = get_stock_data(investment_symbol, investment_date, end_date_investment)
        if not investment_data.empty:
            initial_price = investment_data.iloc[0][investment_symbol]
            current_price = investment_data.iloc[-1][investment_symbol]
        else:
            st.error(f'Invalid investment symbol: {investment_symbol}')
            initial_price = current_price = None
    else:
        # Calculate for cryptos
        investment_data = get_crypto_data(investment_symbol, investment_date, end_date_investment)
        if not investment_data.empty:
            initial_price = investment_data['close'].iloc[0]
            current_price = investment_data['close'].iloc[-1]
        else:
            st.error(f'Invalid investment symbol: {investment_symbol}')
            initial_price = current_price = None
    
    if initial_price is not None and current_price is not None:
        profit_or_loss = current_price - initial_price
        percentage_change = (profit_or_loss / initial_price) * 100

        st.write(f'Invested in {investment_symbol} on {investment_date}')
        st.write(f'Initial Price: {initial_price}')
        st.write(f'Current Price: {current_price}')
        st.write(f'Profit/Loss: {profit_or_loss:.2f}')
        st.write(f'Percentage Change: {percentage_change:.2f}%')

# Fetch data for selected symbols
data = pd.DataFrame()

# Fetch stock data
for symbol in selected_stock_symbols:
    symbol_data = get_stock_data(symbol, start_date, end_date)
    data = pd.concat([data, symbol_data['Adj Close']], axis=1)

# Fetch cryptocurrency data
for symbol in selected_crypto_symbols:
    symbol_data = get_crypto_data(symbol, start_date, end_date)
    if symbol_data is not None:
        data = pd.concat([data, symbol_data['close']], axis=1)

data.columns = selected_stock_symbols + selected_crypto_symbols

st.subheader('Raw Data')
st.write(data)

# Predict stock or cryptocurrency price
st.subheader('Price Prediction')

# Check if any cryptocurrency symbols are selected
if selected_crypto_symbols:
    st.warning("Crypto symbols are selected. Displaying raw data and graph.")
    st.subheader('Raw Data')
    st.write(data[selected_crypto_symbols])
    st.subheader('Price Graph')
    st.line_chart(data[selected_crypto_symbols])
else:
    n_years = st.number_input('Years of Prediction:', 1, 10, 1)
    n_days = n_years * 365
    df = data[selected_stock_symbols]
    df['Prediction'] = df[selected_stock_symbols[0]].shift(-n_days)  # Assume the prediction is based on the first selected symbol

    X = np.array(df.drop(['Prediction'], axis=1))
    X = X[:-n_days]
    y = np.array(df['Prediction'])
    y = y[:-n_days]

    # Check if there's enough data for the split
    if len(X) > int(len(X) * 0.2):  # Check if data size is greater than 20% for testing
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor()
        model.fit(x_train, y_train)

        prediction = model.predict(x_test)
        st.subheader('Model Evaluation')
        mse = mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R2 Score: {r2}')

        # Predict the future price
        predict_df = df[selected_stock_symbols].tail(n_days)
        prediction = model.predict(np.array(predict_df))
        st.subheader('Predicted Price')
        st.write(predict_df)
        st.line_chart(predict_df)
    else:
        st.write("Not enough data to perform the split. Please choose a larger date range.")

# Comparison of selected symbols
st.subheader('Comparison of Selected Symbols')
st.line_chart(data)

# News and Sentiment Analysis
st.subheader('News and Sentiment Analysis')

# Retrieve and display news articles for the selected stock symbols
for symbol in selected_stock_symbols:
    st.write(f'News for {symbol}:')
    news = get_symbol_news(symbol)
    for article in news['articles']:
        st.write(article['title'])
        st.write(f"Published at {article['publishedAt']}")
        st.write(article['description'])
        sentiment = perform_sentiment_analysis(article['title'])
        st.write(f"Sentiment Score: {sentiment['compound']}")

# Retrieve and display news articles for the selected crypto symbols
for symbol in selected_crypto_symbols:
    st.write(f'News for {symbol}:')
    news = get_symbol_news(symbol)
    for article in news['articles']:
        st.write(article['title'])
        st.write(f"Published at {article['publishedAt']}")
        st.write(article['description'])
        sentiment = perform_sentiment_analysis(article['title'])
        st.write(f"Sentiment Score: {sentiment['compound']}")

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title='Stock and Crypto Price Prediction App', page_icon='📈')

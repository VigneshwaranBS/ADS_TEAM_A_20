import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ccxt
from datetime import datetime
import openai_helper

st.set_page_config(page_title='Stock and Crypto Price Prediction App', page_icon='📈')

# Initialize the exchange (e.g., Binance)
exchange = ccxt.binance()


# ... (your code for selecting symbols and other input parameters)

# Add a button to extract financial data using OpenAI
if st.sidebar.button("OpenAI API"):
    st.sidebar.text("Use the text area below to enter your financial article for data extraction.")
    news_article = st.sidebar.text_area("Paste your financial article here", height=300)
    
    if st.sidebar.button("Extract"):
        financial_data_df = openai_helper.extract_financial_data(news_article)
        
# ... (rest of your code for data fetching, prediction, and sentiment analysis)

# In the main content area
col1, col2 = st.columns([3, 2])

# Add your financial data extraction widget here
with col1:
    st.title("Data Extraction Tool using OpenAI API")
    if 'financial_data_df' in locals():
        st.dataframe(financial_data_df)
    
with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)
    # Add other content in col2 as needed
    
    # Add a "Cancel" button to hide the financial data extraction widget
    if st.sidebar.button("Cancel"):
        st.sidebar.text("Extraction cancelled")


# Define a function to fetch stock data
def get_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Define a function to fetch cryptocurrency data
def get_crypto_data(symbol, start, end):
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

# Create a numeric input for the number of years for prediction
n_years = st.sidebar.number_input('Years of Prediction:', 1, 10, 1)
n_months = n_years * 12

# Create investment input fields
st.sidebar.subheader('Investment Profile')
investment_symbol = st.sidebar.selectbox('Select Investment Symbol:', selected_stock_symbols + selected_crypto_symbols)
investment_amount = st.sidebar.number_input('Amount Invested (in dollars):', min_value=10)
investment_date = st.sidebar.date_input('Date of Investment', datetime.now().date())

# Fetch data for selected symbols
data = pd.DataFrame()
crypto_data = pd.DataFrame()

# Fetch stock and cryptocurrency data
for symbol in selected_stock_symbols:
    symbol_data = get_stock_data(symbol, start_date, end_date)
    data = pd.concat([data, symbol_data['Adj Close']], axis=1)
data.columns = selected_stock_symbols

for symbol in selected_crypto_symbols:
    symbol_data = get_crypto_data(symbol, start_date, end_date)
    crypto_data = pd.concat([crypto_data, symbol_data['close']], axis=1)
crypto_data.columns = selected_crypto_symbols

st.subheader('Stock Raw Data')
st.write(data)
st.subheader('Crypto Raw Data')
st.write(crypto_data)


# Predict stock prices
if not data.empty:
    st.subheader('Stock Price Prediction')

    for symbol in selected_stock_symbols:
        df = data[[symbol]]
        df['Prediction'] = df[symbol].shift(-n_months)

        X = np.array(df.drop(['Prediction'], axis=1))
        X = X[:-n_months]
        y = np.array(df['Prediction'])
        y = y[:-n_months]

        if len(X) > int(len(X) * 0.2):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Impute missing values in X_train and X_test
            imputer = SimpleImputer(strategy='mean')
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)

            model = RandomForestRegressor()
            model.fit(x_train, y_train)

            predict_df = df[[symbol]].tail(n_months)
            if 'Prediction' in predict_df.columns:
                predict_df.drop(['Prediction'], axis=1, inplace=True)

            # Impute missing values in prediction data
            predict_df = imputer.transform(predict_df)

            prediction = model.predict(np.array(predict_df))
            prediction_df = pd.DataFrame()
            prediction_df['Predicted Date'] = pd.date_range(start=end_date, periods=n_months, freq='M')
            prediction_df[symbol] = prediction

            # Display the predicted table
            st.subheader(f'Predicted {symbol} Price for Next {n_years} Years')
            st.dataframe(prediction_df)

            # Create a graph with only the predicted values
            st.subheader(f'Predicted {symbol} Price Chart')
            st.line_chart(prediction_df.set_index('Predicted Date')[symbol])


if not crypto_data.empty:
    st.subheader('Crypto Price Prediction')

    for symbol in selected_crypto_symbols:
        df = crypto_data[[symbol]]
        df['Prediction'] = df[symbol].shift(-n_months)

        X = np.array(df.drop(['Prediction'], axis=1))
        X = X[:-n_months]
        y = np.array(df['Prediction'])
        y = y[:-n_months]

        if len(X) > int(len(X) * 0.2):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Impute missing values in X_train and X_test
            imputer = SimpleImputer(strategy='mean')
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)

            model = RandomForestRegressor()
            model.fit(x_train, y_train)

            prediction = model.predict(x_test)

            mse = mean_squared_error(y_test, prediction)
            r2 = r2_score(y_test, prediction)
            st.write(f'Mean Squared Error for {symbol}: {mse}')
            st.write(f'R2 Score for {symbol}: {r2}')

            predict_df = df[[symbol]].tail(n_months)
            if 'Prediction' in predict_df.columns:
                predict_df.drop(['Prediction'], axis=1, inplace=True)

            # Impute missing values in prediction data
            predict_df = imputer.transform(predict_df)

            prediction = model.predict(np.array(predict_df))
            prediction_df = df[[symbol]].tail(n_months).copy()
            prediction_df['Prediction'] = prediction

            prediction_df['Predicted Date'] = pd.date_range(start=end_date, periods=n_months, freq='M')
            prediction_df = prediction_df[['Predicted Date', symbol, 'Prediction']]

            st.subheader(f'Predicted {symbol} Price for Next {n_years} Years')
            st.write(prediction_df)

            st.subheader(f'Current vs. Predicted {symbol} Price')
            st.line_chart(prediction_df.set_index('Predicted Date'))

# If the prediction_df is not defined, display a warning message
if 'prediction_df' in locals() or 'prediction_df' in globals():
    latest_prediction_date = prediction_df['Predicted Date'].max()
else:
    latest_prediction_date = None

if latest_prediction_date:
    st.write(f'Latest Predicted Date: {latest_prediction_date}')
else:
    st.warning("No prediction data available. Please make predictions first.")


latest_prediction_date = prediction_df['Predicted Date'].max()

if investment_symbol and investment_amount and investment_date:
    st.subheader('Investment Profile')

    if investment_date not in data.index and investment_date not in crypto_data.index:
        st.write(f'Investment date is not available in the data. Please select a different date.')
    else:
        if investment_symbol in data.columns:
            investment_data = data[[investment_symbol]].copy()
            current_price = investment_data.loc[latest_prediction_date, investment_symbol]
            initial_price = investment_data.loc[investment_date, investment_symbol]
        elif investment_symbol in crypto_data.columns:
            investment_data = crypto_data[[investment_symbol]].copy()
            current_price = investment_data.loc[latest_prediction_date, investment_symbol]
            initial_price = investment_data.loc[investment_date, investment_symbol]
        else:
            st.write(f'Invalid investment symbol. Please select a different symbol.')

        if initial_price is not None:
            profit_or_loss = current_price - initial_price
            percentage_change = (profit_or_loss / initial_price) * 100

            st.write(f'Invested in {investment_symbol} on {investment_date.date()}')
            st.write(f'Initial Investment Amount: ${investment_amount:.2f}')
            st.write(f'Current Value: ${current_price:.2f}')
            st.write(f'Profit/Loss: ${profit_or_loss:.2f} ({percentage_change:.2f}%)')

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


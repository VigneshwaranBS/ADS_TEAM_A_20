import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ccxt
# import plotly.express as px
import re
from data_processing import preprocess_and_categorize, extract_data_from_pdf
from datetime import datetime
import openai_helper

st.set_page_config(page_title='Stock and Crypto Price Prediction App', page_icon='📈')
st.sidebar.title("Financial Tools")
page = st.sidebar.selectbox("Select a Input", ["Market Analysis", "Financial Data Extraction Page", "Bank Statement"])

exchange = ccxt.binance()

if page == "Financial Data Extraction Page":
    col1, col2 = st.columns([3, 3])

    financial_data_df = pd.DataFrame({
        "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
        "Value": ["", "", "", "", ""]
    })

    with col1:
        st.title("Data Extraction Tool using OpenAI API")
        news_article = st.text_area("Paste your financial article here", height=300)
        if st.button("Extract"):
            financial_data_df = openai_helper.extract_financial_data(news_article)

    with col2:
        st.markdown("<br/>" * 5, unsafe_allow_html=True)
        st.dataframe(financial_data_df.style.set_table_styles([
            dict(selector="th", props=[("font-size", "180%")]),
            dict(selector="td", props=[("font-size", "180%")]),
        ]), width=800)

def get_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

def get_crypto_data(symbol, start, end):
    start_timestamp = pd.Timestamp(start).timestamp() * 1000
    end_timestamp = pd.Timestamp(end).timestamp() * 1000

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=int(start_timestamp), limit=500, params={'endTime': int(end_timestamp)})

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if page == "Market Analysis":
    st.title('Market Analysis')

    def get_symbol_news(symbol):
        newsapi = NewsApiClient(api_key='ef7280046c0d433bbf74a27166e796b5')  
        news = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page=1, page_size=5)
        return news

    def perform_sentiment_analysis(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        return sentiment

    available_stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
    available_crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'DOGE/USDT']
    additional_stock_symbols = ['TSLA', 'GOOG', 'FB', 'NFLX', 'INTC']
    additional_crypto_symbols = ['BNB/USDT', 'ADA/USDT', 'MATIC/USDT', 'DOT/USDT']
    available_stock_symbols += additional_stock_symbols
    available_crypto_symbols += additional_crypto_symbols

    st.sidebar.header('Input Parameters')
    selected_stock_symbols = st.sidebar.multiselect('Select Stock Symbols', available_stock_symbols)
    selected_crypto_symbols = st.sidebar.multiselect('Select Crypto Symbols', available_crypto_symbols)
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2018-01-01'))
    end_date = st.sidebar.date_input('End Date', datetime.today())
    n_years = st.sidebar.number_input('Years of Prediction:', 1, 10, 5, 1)
    n_months = n_years * 12

    data = pd.DataFrame()
    crypto_data = pd.DataFrame()

    for symbol in selected_stock_symbols:
        symbol_data = get_stock_data(symbol, start_date, end_date)
        data = pd.concat([data, symbol_data['Adj Close']], axis=1)
    data.columns = selected_stock_symbols
    for symbol in selected_crypto_symbols:
        symbol_data = get_crypto_data(symbol, start_date, end_date)
        crypto_data = pd.concat([crypto_data, symbol_data['close']], axis=1)
    crypto_data.columns = selected_crypto_symbols

    # Predict stock prices
    if not data.empty:
        st.subheader('Stock Price Prediction')
        st.subheader('Stock Raw Data')
        st.write(data)

        for symbol in selected_stock_symbols:
            df = data[[symbol]]
            df['Prediction'] = df[symbol].shift(-n_months)
            df = df.dropna()

            X = np.array(df.drop(['Prediction'], axis=1))
            X = X[:-n_months]
            y = np.array(df['Prediction'])
            y = y[:-n_months]

            if len(X) > int(len(X) * 0.2):
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                imputer = SimpleImputer(strategy='median')
                x_train = imputer.fit_transform(x_train)
                x_test = imputer.transform(x_test)

                model = RandomForestRegressor()
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)

                mse = mean_squared_error(y_test, y_pred)
                acc = r2_score(y_test, y_pred)

                # acc = accuracy_score(x_test, y_test)
                st.subheader(f'Model Evaluation for {symbol}')
                # st.write('Random Forest')
                st.write(f'Mean Squared Error (MSE): {mse}')
                # st.write(f'R-squared (R2 score): {r2}')
                st.write(f'Accuray (Accuracy_score): {acc*100}')

                predict_df = data[[symbol]].tail(n_months)
                predict_df = predict_df.fillna(df.median())
                predict_df = imputer.transform(predict_df)

                prediction = model.predict(np.array(predict_df))
                prediction_df = pd.DataFrame()
                prediction_df['Predicted Date'] = pd.date_range(start=end_date, periods=n_months, freq='M')
                prediction_df[symbol] = prediction

                # Display the predicted table
                st.subheader(f'Predicted {symbol} Price for Next {n_years} Years')
                st.dataframe(prediction_df)

                st.subheader(f'Predicted {symbol} Price Chart')
                st.line_chart(prediction_df.set_index('Predicted Date')[symbol])

    # Predict crypto prices
    if not crypto_data.empty:
        st.subheader('Crypto Raw Data')
        st.write(crypto_data)

        for symbol in selected_crypto_symbols:  
            df = crypto_data[[symbol]]
            df['Prediction'] = df[symbol].shift(-n_months)
            df = df.dropna()

            X = np.array(df.drop(['Prediction'], axis=1))
            X = X[:-n_months]
            y = np.array(df['Prediction'])
            y = y[:-n_months]

            if len(X) > int(len(X) * 0.2):
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                imputer = SimpleImputer(strategy='median')
                x_train = imputer.fit_transform(x_train)
                x_test = imputer.transform(x_test)

                model = RandomForestRegressor()
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_pred)
                acc = r2_score(y_test, y_pred)

                # Display the model accuracy
                st.subheader(f'Model Accuracy for {symbol}')
                # st.write('Random Forest')
                st.write(f'Mean Squared Error (MSE): {mse}')
                st.write(f'Accuray (Accuracy_score): {acc*100}')

                predict_df = crypto_data[[symbol]].tail(n_months)
                predict_df = predict_df.fillna(df.median())
                predict_df = imputer.transform(predict_df)

                prediction = model.predict(np.array(predict_df))
                prediction_df = pd.DataFrame()
                prediction_df['Predicted Date'] = pd.date_range(start=end_date, periods=n_months, freq='M')
                prediction_df[symbol] = prediction

                # Display the predicted table
                st.subheader(f'Predicted {symbol} Price for Next {n_years} Years')
                st.dataframe(prediction_df)

                st.subheader(f'Predicted {symbol} Price Chart')
                st.line_chart(prediction_df.set_index('Predicted Date')[symbol])

        if 'prediction_df' in locals() or 'prediction_df' in globals():
            latest_prediction_date = prediction_df['Predicted Date'].max()
        else:
            latest_prediction_date = None

        if latest_prediction_date:
            st.write(f'Latest Predicted Date: {latest_prediction_date}')
        else:
            st.caption("No prediction data available. Please make predictions first.")

    # News and Sentiment Analysis
    # st.subheader('News and Sentiment Analysis')

    # Retrieve and display news articles for the selected stock symbols
    for symbol in selected_stock_symbols:
        st.subheader('News and Sentiment Analysis for Stocks')
        st.write(f'News for {symbol}:')
        news = get_symbol_news(symbol)
        for article in news['articles']:
            st.write(article['title'])
            st.write(f"Published at {article['publishedAt']}")
            st.write(article['description'])

            # Perform sentiment analysis
            title_sentiment = perform_sentiment_analysis(article['title'])
            description_sentiment = perform_sentiment_analysis(article['description'])

            st.write(f"Title Sentiment Score: {title_sentiment['compound']}")
            st.write(f"Description Sentiment Score: {description_sentiment['compound']}")

    for symbol in selected_crypto_symbols:
        st.subheader('News and Sentiment Analysis for Crypto') 
        st.write(f'News for {symbol}:')
        news = get_symbol_news(symbol)
        for article in news['articles']:
            st.write(article['title'])
            st.write(f"Published at {article['publishedAt']}")
            st.write(article['description'])
            sentiment = perform_sentiment_analysis(article['title'])
            st.write(f"Sentiment Score: {sentiment['compound']}") 

# Bank Statement Page
if page == "Bank Statement":
    st.title("Bank Statement Analysis")
    
    # Function to extract and format the date from descriptions
    def extract_and_format_date(description):
        # Regular expression pattern to match a date in the format "YYYY-MM-DD"
        date_pattern = r'\d{2}-\d{2}-\d{4}'
        
        # Find the first date in the description
        match = re.search(date_pattern, description)
        
        if match:
            date = match.group()
            return date
        
        # Return a default date if no date is found
        return "N/A"

    # Function to determine the type of transaction (income or expense)
    def transaction_type(amount):
        if amount >= 0:
            return 'Income'
        else:
            return 'Expense'

    st.sidebar.subheader('Upload Your Financial Data (PDF only)')
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_data = extract_data_from_pdf(uploaded_file)
        df = preprocess_and_categorize(pdf_data)

        df['Bank Balance'] = df['Amount'].cumsum()

        df['Transaction Type'] = df['Amount'].apply(transaction_type)

        df['Income Balance'] = df[df['Transaction Type'] == 'Income']['Amount'].cumsum()
        df['Expense Balance'] = df[df['Transaction Type'] == 'Expense']['Amount'].cumsum()

        df['Net Bank Balance'] = df['Income Balance'] - df['Expense Balance']

        df['Date'] = df['Description'].apply(extract_and_format_date)

        df = df.drop(columns=['Description'])

        higher_expenses = df[df['Amount'] > 300]
        lower_spendings = df[df['Amount'] < 300]

        higher_expenses['Type'] = 'Higher Expenses'
        lower_spendings['Type'] = 'Lower Spendings'
        combined_df = pd.concat([higher_expenses, lower_spendings])

        line_fig = px.line(combined_df, x='Date', y='Amount', labels={'Amount': 'Expense Amount'}, 
                        title='Expense Over Time for Higher Expenses and Lower Spendings', color='Type')

        balance_fig = px.line(df, x='Date', y=['Net Bank Balance', 'Income Balance', 'Expense Balance'], 
                            labels={'Net Bank Balance': 'Net Balance', 'Income Balance': 'Income Balance', 'Expense Balance': 'Expense Balance'},
                            title='Bank Balance Over Time')

        selected_option = st.sidebar.selectbox("Select an option:", ("All Transactions", "Higher Expenses", "Lower Spendings"))
        
        selected_graph = st.sidebar.selectbox("Select a graph:", ("Expense Over Time", "Bank Balance Over Time"))

        if selected_option == "Higher Expenses":
            st.subheader('Higher Expenses')
            st.write(higher_expenses)
        elif selected_option == "Lower Spendings":
            st.subheader('Lower Spendings')
            st.write(lower_spendings)
        else:
            st.subheader('All Transactions and Balances')
            st.write(df)



        if selected_graph == "Expense Over Time":
            st.subheader('Expense Over Time')
            st.plotly_chart(line_fig)
        else:
            st.subheader('Bank Balance Over Time')
            st.plotly_chart(balance_fig)



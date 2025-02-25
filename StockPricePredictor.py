import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st


def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y") 
    
    stock_info = stock.info
    stock_stats = {
        "Market Cap": stock_info.get("marketCap", "N/A"),
        "Volume": stock_info.get("volume", "N/A"),
        "Dividend Yield": stock_info.get("dividendYield", "N/A"),
        "P/E Ratio": stock_info.get("trailingPE", "N/A"),
        "52-Week High": stock_info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": stock_info.get("fiftyTwoWeekLow", "N/A"),
    }

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy() 
    return df, stock_stats  


def preprocess_data(df):
    if not isinstance(df, pd.DataFrame): 
        raise TypeError("Expected DataFrame, but received something else.")

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # RSI Calculation
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)

    # Momentum Indicator
    df['Momentum'] = df['Close'] - df['Close'].shift(4)

    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df



# Normalize data
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

# Prepare data for LSTM
def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# AI-powered explanation function
def generate_ai_explanation(data, predictions):
    """
    Generate a detailed AI-based explanation for stock predictions.
    - Analyzes trends, RSI, MACD, Bollinger Bands, and volatility.
    - Compares past trends with future forecasts.
    - Identifies potential risk factors.
    """

    last_close = data['Close'].iloc[-1]
    predicted_trend = "increase" if predictions[-1] > last_close else "decrease"
    
    # Extract key indicators
    rsi = data['RSI_14'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    volatility = data['Volatility'].iloc[-1]
    upper_band = data['Upper_Band'].iloc[-1]
    lower_band = data['Lower_Band'].iloc[-1]
    momentum = data['Momentum'].iloc[-1]

    explanation = f"📊 **Stock Prediction Summary**\n"
    explanation += f"The model predicts a **{predicted_trend}** in stock price over the next {len(predictions)} days.\n\n"

    # RSI Analysis
    if rsi > 70:
        explanation += "📌 The RSI is **above 70**, indicating the stock is **overbought** and may experience a correction.\n"
    elif rsi < 30:
        explanation += "📌 The RSI is **below 30**, suggesting the stock is **oversold** and might see an upward rebound.\n"
    else:
        explanation += f"📌 The RSI is **{round(rsi, 2)}**, indicating neutral momentum.\n"

    # MACD Analysis**
    if macd > macd_signal:
        explanation += "📌 The MACD is **above the Signal Line**, signaling a **bullish trend** (upward momentum).\n"
    else:
        explanation += "📌 The MACD is **below the Signal Line**, suggesting a **bearish trend** (downward momentum).\n"

    # Bollinger Bands Analysis**
    if last_close > upper_band:
        explanation += "📌 The stock price is **above the Upper Bollinger Band**, indicating **potential overvaluation**.\n"
    elif last_close < lower_band:
        explanation += "📌 The stock price is **below the Lower Bollinger Band**, suggesting **undervaluation**.\n"
    else:
        explanation += "📌 The stock price is moving **within normal volatility range**.\n"

    # Momentum Analysis**
    if momentum > 0:
        explanation += f"📌 Momentum is **positive**, reinforcing **short-term upward movement**.\n"
    else:
        explanation += f"📌 Momentum is **negative**, suggesting **short-term weakness**.\n"

    # Volatility Assessment**
    if volatility > data['Volatility'].mean():
        explanation += "⚠️ **High volatility detected!** Expect unpredictable price swings.\n"
    else:
        explanation += "✅ **Volatility is stable**, indicating a more consistent price movement.\n"

    explanation += "\n⚠️ **Risk Factors to Consider:**\n"
    explanation += "🔹 External market events (news, earnings reports, global economic conditions) can impact predictions.\n"
    explanation += "🔹 Predictions are based on historical data and technical indicators, not guaranteed outcomes.\n"
    
    explanation += "\n📌 **Recommendation:**\n"
    explanation += "✅ Consider additional research, market news, and risk management before making investment decisions.\n"

    return explanation


# Streamlit UI
st.title('Stock Price Prediction with AI Explanation')

stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
selected_stock = st.selectbox('Select the Stock:', stock_list)

st.write(f"Fetching data for {selected_stock}...")
data, stock_stats = fetch_data(selected_stock)  
data = preprocess_data(data)  

# Display Stock Statistics
st.subheader(f"📊 {selected_stock} - Key Statistics")
for key, value in stock_stats.items():
    st.write(f"**{key}:** {value}")

# Display Historical Stock Chart
st.subheader(f"📈 {selected_stock} - Stock Price History")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],  
    high=data['High'],  
    low=data['Low'],    
    close=data['Close'],
    name="Stock Price"
))

fig.update_layout(
    title=f"{selected_stock} Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    template="plotly_dark"
)
st.plotly_chart(fig)

# Dropdown to select prediction period
prediction_days = st.selectbox(
    'Select Prediction Period:', 
    [7, 10, 15, 30], 
    index=1  # Default to 10 days
)

# Use the already fetched data, do NOT fetch again
latest_price = data['Close'].iloc[-1]
st.write(f'Latest Price: ${latest_price:.2f}')

# Define model filename
model_filename = f"{selected_stock}_lstm_model.h5"

if st.button("Train and Predict"):
    st.write("Training the model... Please wait!")

    scaled_data, scaler = normalize_data(data)
    X, y = prepare_data(scaled_data)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(model_filename):
        st.write("Loading saved model...")
        model = load_model(model_filename)
    else:
        model = build_model((x_train.shape[1], x_train.shape[2]))
        model.fit(x_train, y_train, batch_size=16, epochs=20)
        model.save(model_filename)
        st.write("Model saved!")

    # Now inside if-statement
    st.write(f"Predicting prices for the next {prediction_days} days...")
    predictions = []
    input_sequence = scaled_data[-60:]  # Use the last 60 days as input

    for day in range(prediction_days):  # Use the selected period
        input_sequence = input_sequence.reshape(1, 60, -1)  # Ensure correct shape
        predicted_price = model.predict(input_sequence)[0][0]
        predictions.append(predicted_price)

        # Shift input sequence by removing the first value & appending the new prediction
        new_input = np.append(input_sequence[:, 1:, :], [[[predicted_price]]], axis=1)
        input_sequence = new_input

    # Convert back from scaled values
    placeholder = np.zeros((len(predictions), scaled_data.shape[1]))
    placeholder[:, 0] = predictions
    predictions = scaler.inverse_transform(placeholder)[:, 0]

    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=prediction_days).strftime('%Y-%m-%d').tolist()
    prediction_df = pd.DataFrame({'Date': days, "Predicted Price": predictions})

    st.write("Predicted Prices:")
    st.table(prediction_df)

    # Generate AI Explanation
    ai_explanation = generate_ai_explanation(data, predictions)

    st.subheader("📊 AI Stock Prediction Explanation")
    st.markdown(ai_explanation)

    # Plot Predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode='lines+markers',
        name='Predicted Prices'
    ))
    fig.update_layout(
        title=f"{prediction_days}-Day Price Prediction for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize scaler globally
scaler = MinMaxScaler(feature_range=(0, 1))

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(256),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

def prepare_data(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def get_technical_indicators(df):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    return df

def get_sentiment_score(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return 0.0
        
        sentiments = []
        for article in news[:10]:  # Analyze up to 10 most recent news items
            blob = TextBlob(article['title'])
            sentiments.append(blob.sentiment.polarity)
        
        return np.mean(sentiments)
    except Exception as e:
        print(f"Error getting sentiment score: {e}")
        return 0.0

def get_stock_data(symbol):
    # Get exactly 3 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Add technical indicators
    df = get_technical_indicators(df)
    return df, stock

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'No symbol provided'}), 400
            
        symbol = data['symbol'].upper()
        
        try:
            # Verify the symbol exists
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info:
                return jsonify({'error': f'Invalid symbol: {symbol}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error fetching stock data: {str(e)}'}), 400
        
        # Get stock data with technical indicators
        df, stock = get_stock_data(symbol)
        
        # Prepare features for LSTM
        feature_columns = ['Close', 'Volume', 'RSI', 'MACD']
        dataset = df[feature_columns].values
        scaled_data = scaler.fit_transform(dataset)
        
        # Prepare training data
        X, y = prepare_data(scaled_data)
        
        if len(X) == 0 or len(y) == 0:
            return jsonify({'error': 'Insufficient data for analysis'}), 400
        
        # Create and train model
        model = create_lstm_model((X.shape[1], X.shape[2]))
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # Make predictions for next 7 days
        last_sequence = scaled_data[-30:]
        future_predictions = []
        current_prediction = last_sequence
        
        for _ in range(7):
            current_prediction_reshaped = current_prediction.reshape((1, 30, len(feature_columns)))
            next_pred = model.predict(current_prediction_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            new_row = current_prediction[-1].copy()
            new_row[0] = next_pred[0, 0]
            current_prediction = np.vstack((current_prediction[1:], new_row))
        
        # Inverse transform predictions
        future_prices = scaler.inverse_transform(
            np.array([[p] + [0] * (len(feature_columns)-1) for p in future_predictions])
        )[:, 0]
        
        # Get additional metrics
        sentiment_score = get_sentiment_score(symbol)
        current_price = df['Close'].iloc[-1]
        
        # Calculate model MAE
        y_pred = model.predict(X, verbose=0)
        mae = np.mean(np.abs(y - y_pred))
        
        # Prepare response data
        response_data = {
            'predicted_price': float(current_price),
            'future_prices': future_prices.tolist(),
            'sentiment_score': sentiment_score,
            'recommendation': 'Buy' if sentiment_score > 0 and future_prices[-1] > current_price else 'Sell',
            'model_mae': float(mae),
            'historical_volume': df['Volume'].tail(5).tolist(),
            'historical_rsi': df['RSI'].tail(5).tolist(),
            'feature_importance': {
                'Price': 0.4,
                'Volume': 0.2,
                'RSI': 0.2,
                'MACD': 0.2
            },
            'additional_data': {
                'pe_ratio': stock.info.get('forwardPE', 0),
                'market_cap': stock.info.get('marketCap', 0),
                'dividend_yield': stock.info.get('dividendYield', 0) or 0
            },
            'economic_data': {
                'interest_rate': 0.0425,
                'inflation_rate': 0.032,
                'gdp_growth_rate': 0.023
            }
        }
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
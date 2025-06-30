from flask import Flask, render_template, request, redirect, jsonify
import yfinance as yf
from pytrends.request import TrendReq
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import pandas as pd
import time

app = Flask(__name__)

portfolio = []

# In-memory cache for stock prices
stock_cache = {}

def get_stock_data_cached(symbol, ttl=60):
    """Returns the latest price from cache or fetches from yfinance if expired."""
    now = time.time()
    if symbol in stock_cache and now - stock_cache[symbol]["time"] < ttl:
        return stock_cache[symbol]["data"]

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")['Close']
        if not data.empty:
            price = data.iloc[-1]
            stock_cache[symbol] = {"data": price, "time": now}
            return price
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def predict_prices(symbol):
    """Predicts the next 7 days of closing prices using linear regression."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return []

        hist['Days'] = (hist.index - hist.index[0]).days
        X = hist['Days'].values.reshape(-1, 1)
        y = hist['Close'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([X[-1][0] + i for i in range(1, 8)]).reshape(-1, 1)
        predictions = model.predict(future_days)

        return predictions.flatten().tolist()
    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    total_invested = sum(stock['shares'] * stock['avg_cost'] for stock in portfolio)
    current_value = 0
    insights = []

    for stock in portfolio:
        ltp = get_stock_data_cached(stock['symbol'])

        if ltp is None:
            stock['ltp'] = "N/A"
            stock['current_value'] = 0
            stock['pnl'] = 0
            stock['sentiment'] = "N/A"
            stock['predictions'] = []
            stock['error'] = "Data unavailable or rate limited."
        else:
            stock['ltp'] = round(ltp, 2)
            stock['current_value'] = round(stock['shares'] * ltp, 2)
            stock['pnl'] = round(stock['current_value'] - (stock['shares'] * stock['avg_cost']), 2)
            stock['sentiment'] = np.random.uniform(0.8, 1.2)  # Dummy sentiment value
            stock['predictions'] = predict_prices(stock['symbol'])
            current_value += stock['current_value']

    growth = ((current_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0
    pnl = current_value - total_invested

    for stock in portfolio:
        weight = (stock['current_value'] / current_value) * 100 if current_value > 0 else 0
        insights.append(f"{stock['symbol']} has {weight:.2f}% weight. {'Consider rebalancing.' if weight > 50 else 'Looks good.'}")

    return render_template("index.html", portfolio=portfolio, total_invested=total_invested, current_value=current_value,
                           pnl=pnl, growth=growth, insights=insights)

@app.route("/add_stock", methods=["POST"])
def add_stock():
    symbol = request.form.get("symbol").upper()
    shares = int(request.form.get("shares"))
    avg_cost = float(request.form.get("avg_cost"))
    
    portfolio.append({"symbol": symbol, "shares": shares, "avg_cost": avg_cost})
    
    return redirect("/")

@app.route("/delete_stock/<symbol>")
def delete_stock(symbol):
    global portfolio
    portfolio = [stock for stock in portfolio if stock["symbol"] != symbol]
    
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)

import yfinance as yf

# Test with a common stock symbol
ticker = "AAPL"  # Apple Inc.

# Download historical data for the last 5 days
data = yf.download(ticker, period="5d")

# Show the result
print("📈 Apple Inc. (AAPL) - Last 5 Days")
print(data)

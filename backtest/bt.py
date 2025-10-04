import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from bokeh.io import export_png
import os

# Load your historical data (replace with your CSV path)
data = pd.read_csv('btc_data.csv', parse_dates=['timestamp'], index_col='timestamp', sep=';')

# Ensure required columns are present
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV missing required columns: {required_columns}")

# Select only necessary columns
data = data[required_columns]

# Sort the index to ensure chronological order
data.sort_index(ascending=True, inplace=True)

# Resample to daily frequency to ensure consistent timestamps
data = data.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Verify data is not empty
if data.empty:
    raise ValueError("Loaded DataFrame is empty. Check CSV file or parsing.")

class MovingAverageCrossover(Strategy):
    short_ma = 50  # Short moving average period
    long_ma = 200  # Long moving average period

    def init(self):
        # Define a function to compute rolling mean
        def rolling_mean(series, window):
            return pd.Series(series).rolling(window=window, min_periods=1).mean().values

        # Apply rolling mean to Close price using self.I
        self.ma_short = self.I(rolling_mean, self.data.Close, self.short_ma)
        self.ma_long = self.I(rolling_mean, self.data.Close, self.long_ma)

    def next(self):
        if crossover(self.ma_short, self.ma_long):
            self.buy()  # Enter long position
        elif crossover(self.ma_long, self.ma_short):
            self.sell()  # Exit or go short (if allowed; adjust for long-only)

# Run the backtest
try:
    bt = Backtest(data, MovingAverageCrossover, cash=1000000, commission=0.001)  # Starting cash, trading fee
    stats = bt.run()
except Exception as e:
    print(f"Backtest failed: {e}")
    exit(1)

# Print results
print(stats)

# Generate and save the plot as a PNG
try:
    plot = bt.plot(open_browser=False, superimpose=False)  # Disable superimpose to avoid upsampling error
    export_png(plot, filename="backtest_plot.png")  # Save plot to file
    print("Plot saved as 'backtest_plot.png'")
except Exception as e:
    print(f"Failed to save plot: {e}")
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Black-Scholes formula for European call option price
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

class BitcoinCoveredCallsBacktester:
    """
    Backtester for selling covered calls on Bitcoin, aiming to avoid having BTC called away.

    Features:
    - Sells OTM calls at specified strike multipliers (e.g., 1.10 = 10% OTM).
    - Tests multiple expiration durations.
    - Optional: Only sell when volatility > threshold for higher premiums.
    - Tracks exercise frequency to assess risk of losing BTC.
    - Compares to buy-and-hold BTC.
    - Assumes continuous BTC holding (additive PnL model).
    """

    def __init__(self, start_date='2020-01-01', end_date=None, r=0.05, initial_capital=10000, roll_frequency_days=30, min_vol=0.0):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.r = r  # Risk-free rate
        self.initial_capital = initial_capital
        self.roll_frequency_days = roll_frequency_days  # How often to sell new calls
        self.min_vol = min_vol  # Minimum volatility to sell calls (0 to disable)
        self.data = None
        self.signals = None

    def fetch_data(self):
        """Fetch historical BTC-USD data using yfinance."""
        ticker = 'BTC-USD'
        self.data = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        if self.data.empty:
            raise ValueError(f"No data fetched for {ticker} from {self.start_date} to {self.end_date}")
        # Ensure unique index
        self.data = self.data.loc[~self.data.index.duplicated(keep='first')]
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=30).std() * np.sqrt(252)
        # Fill NaN volatility with 0 for early periods
        self.data['Volatility'] = self.data['Volatility'].fillna(0)
        print(f"Data fetched: {len(self.data)} rows from {self.data.index[0]} to {self.data.index[-1]}")

    def generate_signals(self):
        """
        Generate signals for selling calls (1=sell, 0=hold).
        Default: Sell every roll_frequency_days if volatility > min_vol.
        """
        if self.data is None:
            raise ValueError("Fetch data first!")

        # Ensure index is unique
        if not self.data.index.is_unique:
            self.data = self.data.loc[~self.data.index.duplicated(keep='first')]

        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        roll_dates = [start_dt + timedelta(days=i) for i in range(0, (end_dt - start_dt).days + 1, self.roll_frequency_days)]
        roll_dates = [d for d in roll_dates if d <= end_dt]

        signals = pd.DataFrame(index=self.data.index, data={'signal': 0})
        for roll_date in roll_dates:
            try:
                next_date = signals.index[signals.index >= pd.Timestamp(roll_date)][0]
                vol = self.data.loc[next_date, 'Volatility'].item() if isinstance(self.data.loc[next_date, 'Volatility'], pd.Series) else self.data.loc[next_date, 'Volatility']
                if pd.notna(vol) and vol >= self.min_vol:
                    signals.loc[next_date, 'signal'] = 1
            except (IndexError, KeyError, ValueError):
                continue  # Skip if date or volatility data is unavailable or invalid

        self.signals = signals
        print(f"Signals generated: {self.signals['signal'].sum()} calls sold (every {self.roll_frequency_days} days, vol > {self.min_vol:.2f}).")

    def simulate_strategy(self, strike_multiplier=1.10, expiry_days=30):
        """
        Simulate covered call strategy for given strike multiplier and expiry days.
        Returns trades and portfolio DataFrames.
        """
        if self.signals is None:
            self.generate_signals()

        btc_held = 1  # Normalize to 1 BTC
        first_date = self.data.index[0]
        btc_price_initial = self.data.loc[first_date, 'Close'].item() if isinstance(self.data.loc[first_date, 'Close'], pd.Series) else self.data.loc[first_date, 'Close']
        scale_factor = self.initial_capital / btc_price_initial
        cumulative_option_pnl = 0
        trades = []
        buy_and_hold = []
        portfolio = []
        active_position = None

        for date in self.data.index:
            S = self.data.loc[date, 'Close'].item() if isinstance(self.data.loc[date, 'Close'], pd.Series) else self.data.loc[date, 'Close']
            sigma = self.data.loc[date, 'Volatility'].item() if isinstance(self.data.loc[date, 'Volatility'], pd.Series) else self.data.loc[date, 'Volatility']

            # Update portfolio values
            current_buy_and_hold = btc_held * S
            current_portfolio = current_buy_and_hold + cumulative_option_pnl
            buy_and_hold.append((date, current_buy_and_hold * scale_factor))
            portfolio.append((date, current_portfolio * scale_factor))

            # Handle expiry
            if active_position and date >= active_position['expiry_date']:
                S_expiry = S
                K = active_position['K']
                premium = active_position['premium']
                option_pnl = premium - max(S_expiry - K, 0)
                exercised = S_expiry > K
                cumulative_option_pnl += option_pnl

                trades.append({
                    'entry_date': active_position['entry_date'],
                    'expiry_date': date,
                    'entry_price': active_position['S_entry'],
                    'strike': K,
                    'premium': premium * scale_factor,
                    'expiry_price': S_expiry,
                    'option_pnl': option_pnl * scale_factor,
                    'cumulative_option_pnl': cumulative_option_pnl * scale_factor,
                    'exercised': exercised
                })
                active_position = None

            # Sell new call
            if self.signals.loc[date, 'signal'] == 1 and active_position is None and pd.notna(sigma) and sigma > 0:
                T = expiry_days / 365.0
                K = S * strike_multiplier
                premium = black_scholes_call(S, K, T, self.r, sigma)

                expiry_dt = date + timedelta(days=expiry_days)
                if expiry_dt > self.data.index[-1]:
                    continue
                try:
                    expiry_date = self.data.index[self.data.index >= expiry_dt][0] if expiry_dt not in self.data.index else expiry_dt
                except IndexError:
                    continue  # Skip if expiry date is beyond data

                active_position = {
                    'entry_date': date,
                    'expiry_date': expiry_date,
                    'S_entry': S,
                    'K': K,
                    'premium': premium
                }

        self.buy_and_hold = pd.DataFrame(buy_and_hold, columns=['Date', 'BuyAndHold']).set_index('Date')
        self.portfolio = pd.DataFrame(portfolio, columns=['Date', 'Portfolio']).set_index('Date')
        self.trades = pd.DataFrame(trades)
        return self.trades, self.portfolio

    def run_backtests(self, strike_multipliers=[1.05, 1.10, 1.20], expiry_days_list=[7, 14, 30]):
        """
        Run backtests for combinations of strike multipliers and expiry days.
        Returns summary DataFrame with exercise frequency and other metrics.
        """
        results = []
        for strike_mult in strike_multipliers:
            for expiry_days in expiry_days_list:
                trades, portfolio = self.simulate_strategy(strike_mult, expiry_days)

                if portfolio.empty:
                    continue

                total_return = (portfolio['Portfolio'].iloc[-1] / self.initial_capital - 1) * 100
                bah_return = (self.buy_and_hold['BuyAndHold'].iloc[-1] / self.initial_capital - 1) * 100
                num_trades = len(trades)
                avg_premium = trades['premium'].mean() if num_trades > 0 else 0
                win_rate = (trades['option_pnl'] > 0).mean() * 100 if num_trades > 0 else 0
                exercise_rate = trades['exercised'].mean() * 100 if num_trades > 0 else 0

                results.append({
                    'Strike_Mult': strike_mult,
                    'Expiry_Days': expiry_days,
                    'Total_Return_%': total_return,
                    'BuyAndHold_Return_%': bah_return,
                    'Alpha_%': total_return - bah_return,
                    'Num_Trades': num_trades,
                    'Avg_Premium': avg_premium,
                    'Option_Win_Rate_%': win_rate,
                    'Exercise_Rate_%': exercise_rate
                })

        summary = pd.DataFrame(results)
        print("\nBacktest Summary (Minimized Exercise Risk):")
        print(summary[['Strike_Mult', 'Expiry_Days', 'Total_Return_%', 'Alpha_%', 'Num_Trades', 'Avg_Premium', 'Exercise_Rate_%']].to_string(index=False))

        return summary

    def plot_results(self):
        """Plot portfolio vs buy-and-hold."""
        if self.portfolio is None:
            raise ValueError("Run simulation first!")

        plt.figure(figsize=(12, 6))
        self.portfolio['Portfolio'].plot(label='Covered Calls Portfolio')
        self.buy_and_hold['BuyAndHold'].plot(label='Buy and Hold BTC')
        plt.title('Covered Calls vs Buy-and-Hold')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    backtester = BitcoinCoveredCallsBacktester(
        start_date='2020-01-01',  # Ensure full date range
        roll_frequency_days=30,  # Monthly rolls
        initial_capital=10000,
        min_vol=0.3  # Only sell when 30-day vol > 30%
    )

    backtester.fetch_data()

    # Test OTM strikes and expirations
    summary = backtester.run_backtests(
        strike_multipliers=[1.05, 1.10, 1.20],  # 5%, 10%, 20% OTM
        expiry_days_list=[7, 14, 30]  # Weekly, bi-weekly, monthly
    )

    # Plot a single scenario
    trades, portfolio = backtester.simulate_strategy(strike_multiplier=1.20, expiry_days=14)
    backtester.plot_results()

    # Notes:
    # - Higher strike_multipliers (e.g., 1.20) reduce exercise risk.
    # - Shorter expiry_days reduce time for price to reach strike.
    # - min_vol=0.3 targets high-vol periods for better premiums.
    # - For real data, use Deribit/Tardis.dev historical options data.
    # - Install: pip install yfinance pandas numpy scipy matplotlib
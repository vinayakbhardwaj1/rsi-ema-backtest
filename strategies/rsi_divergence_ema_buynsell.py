import pandas as pd
import requests
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

# ------------------------------------------
# Data Fetching
# ------------------------------------------
def fetch_data(symbol="BTC/USD", interval="1h", api_key=None):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=3000&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'values' not in data:
        raise Exception(f"Error fetching data: {data.get('message', 'Unknown error')}")
    df = pd.DataFrame(data['values'])
    df = df.rename(columns={'datetime': 'date', 'close': 'close'})
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    df = df.sort_values('date').reset_index(drop=True)
    return df

# ------------------------------------------
# Indicators
# ------------------------------------------
def calculate_indicators(df, rsi_period=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df

# ------------------------------------------
# Detect Divergences
# ------------------------------------------
def find_divergence_signals(df):
    signals = []
    for i in range(2, len(df) - 1):
        # Bullish Divergence
        if df['RSI'].iloc[i - 2] < df['RSI'].iloc[i] < 35:
            if df['close'].iloc[i] > df['EMA_50'].iloc[i] > df['EMA_200'].iloc[i]:
                signals.append((df['date'].iloc[i], 'BUY'))

        # Bearish Divergence
        if df['RSI'].iloc[i - 2] > df['RSI'].iloc[i] > 65:
            if df['close'].iloc[i] < df['EMA_50'].iloc[i] < df['EMA_200'].iloc[i]:
                signals.append((df['date'].iloc[i], 'SELL'))

    return signals

# ------------------------------------------
# Backtest with TP/SL for BUY + SELL
# ------------------------------------------
def calculate_max_drawdown(pnl_list):
    equity = [0]
    for p in pnl_list:
        equity.append(equity[-1] + p)
    peak = equity[0]
    drawdowns = [peak - x if x < peak else 0 for x in equity]
    return max(drawdowns)

def backtest_with_tp_sl(df, signals, tp_pct=0.02, sl_pct=0.01, max_hold=10):
    trades = []
    pnl = []

    for time, direction in signals:
        entry_index = df.index[df['date'] == time].tolist()
        if not entry_index:
            continue
        entry_index = entry_index[0]
        entry_price = df.at[entry_index, 'close']
        tp_price = entry_price * (1 + tp_pct) if direction == 'BUY' else entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 - sl_pct) if direction == 'BUY' else entry_price * (1 + sl_pct)

        exit_price = None
        outcome = None

        for j in range(1, max_hold + 1):
            if entry_index + j >= len(df):
                break
            current_price = df.at[entry_index + j, 'close']
            if direction == 'BUY':
                if current_price >= tp_price:
                    exit_price = tp_price
                    outcome = 'TP'
                    break
                elif current_price <= sl_price:
                    exit_price = sl_price
                    outcome = 'SL'
                    break
            else:  # SELL
                if current_price <= tp_price:
                    exit_price = tp_price
                    outcome = 'TP'
                    break
                elif current_price >= sl_price:
                    exit_price = sl_price
                    outcome = 'SL'
                    break

        if not exit_price:
            exit_price = df.at[min(entry_index + max_hold, len(df)-1), 'close']
            outcome = 'TimeExit'

        profit = exit_price - entry_price if direction == 'BUY' else entry_price - exit_price
        pnl.append(profit)
        trades.append((time, entry_price, exit_price, profit, outcome, direction))

    if not pnl:
        print("\nNo trades executed.")
        return []

    win_rate = sum(1 for p in pnl if p > 0) / len(pnl) * 100
    total_pnl = sum(pnl)
    max_dd = calculate_max_drawdown(pnl)

    print("\nBacktest with TP/SL:")
    print(f"Total Trades: {len(pnl)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")

    for t in trades:
        print(f"\n{t[5]} @ {t[0]}: Entry = {t[1]:.2f}, Exit = {t[2]:.2f}, PnL = {t[3]:.2f}, Outcome = {t[4]}")
    return trades

# ------------------------------------------
# Chart
# ------------------------------------------
def plot_strategy(df, signals, trades):
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['date'], df['close'], label='Close Price', color='blue', linewidth=1)
    ax1.plot(df['date'], df['EMA_50'], label='EMA 50', color='orange', linestyle='--')
    ax1.plot(df['date'], df['EMA_200'], label='EMA 200', color='red', linestyle='--')

    for time, signal in signals:
        price = df[df['date'] == time]['close'].values[0]
        if signal == "BUY":
            ax1.scatter(time, price, marker='^', color='green', s=100, label='BUY')
        elif signal == "SELL":
            ax1.scatter(time, price, marker='v', color='red', s=100, label='SELL')

    for trade in trades:
        exit_time = trade[0]
        price = trade[2]
        color = 'green' if trade[5] == 'BUY' else 'red'
        ax1.scatter(exit_time, price, marker='o', color=color, s=50, edgecolors='black', label='Exit')

    ax1.set_title("BTC/USD - RSI Divergence + EMA Strategy")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df['date'], df['RSI'], label='RSI 14', color='purple')
    ax2.axhline(70, linestyle='--', color='red')
    ax2.axhline(30, linestyle='--', color='green')
    ax2.set_title("RSI (14)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    with open("config/api_key.txt", "r") as f:
        api_key = f.read().strip()
    print("API Key Loaded")

    symbol = "BTC/USD"
    df = fetch_data(symbol=symbol, api_key=api_key)
    df = calculate_indicators(df)
    signals = find_divergence_signals(df)
    print(f"\nTotal signals: {len(signals)}")
    for sig in signals[-5:]:
        print(sig)

    trades = backtest_with_tp_sl(df, signals, tp_pct=0.02, sl_pct=0.01, max_hold=10)
    if trades:
        plot_strategy(df, signals, trades)

# ------------------------------------------
# Entry Point
# ------------------------------------------
if __name__ == "__main__":
    main()
    
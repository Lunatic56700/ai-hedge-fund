#!/usr/bin/env python3
"""
TSLA Option Chain Advisor (smart version)

- Fetches nearest expiries for TSLA via yfinance
- Computes trend using SMA10/SMA30, RSI, MACD, and historical volatility
- Selects near-the-money/high-interest strikes and prints smarter advice
- Writes advice to tsla_option_advice.csv
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# --- Config ---
TICKER = "TSLA"
N_EXPIRIES = 3            # how many upcoming expiries to analyze
SMA_SHORT = 10
SMA_LONG = 30
RSI_PERIOD = 14
STRIKE_WINDOW_PCT = 0.30  # include strikes within +/- this % of current price
MAX_PER_SIDE = 10         # limit how many calls/puts per expiry to print
OUTPUT_CSV = "tsla_option_advice.csv"

# --- Helpers ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_trend_and_price(ticker_obj):
    hist = ticker_obj.history(period="6mo", interval="1d", actions=False)
    if hist.empty or 'Close' not in hist.columns:
        raise RuntimeError("No price history returned for TSLA")
    hist = hist.dropna(subset=['Close'])
    
    hist['SMA_short'] = hist['Close'].rolling(SMA_SHORT).mean()
    hist['SMA_long'] = hist['Close'].rolling(SMA_LONG).mean()
    hist['RSI'] = compute_rsi(hist['Close'], RSI_PERIOD)
    hist['MACD'], hist['MACD_signal'] = compute_macd(hist['Close'])
    hist['Volatility'] = hist['Close'].pct_change().rolling(20).std() * np.sqrt(252)  # annualized
    
    last = hist.iloc[-1]
    price = float(last['Close'])
    sma_short = float(last['SMA_short']) if not np.isnan(last['SMA_short']) else None
    sma_long = float(last['SMA_long']) if not np.isnan(last['SMA_long']) else None
    rsi = float(last['RSI']) if not np.isnan(last['RSI']) else None
    macd = float(last['MACD'])
    macd_signal = float(last['MACD_signal'])
    vol = float(last['Volatility'])
    
    # Determine trend using SMA, RSI, MACD
    trend = "neutral"
    if sma_short and sma_long:
        if sma_short > sma_long:
            trend = "bullish"
        elif sma_short < sma_long:
            trend = "bearish"
    # refine trend with MACD + RSI
    if macd > macd_signal and rsi and rsi > 55:
        trend = "bullish"
    elif macd < macd_signal and rsi and rsi < 45:
        trend = "bearish"
    
    return price, trend, sma_short, sma_long, rsi, macd, macd_signal, vol

def days_to(expiry_str):
    exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
    delta = (exp_dt - datetime.now()).days
    return max(delta, 0)

def classify_volatility(vol):
    if vol < 0.25:
        return "low"
    elif vol < 0.50:
        return "moderate"
    else:
        return "high"

def advice_for_option(row, underlying_price, trend, vol_class):
    strike = float(row['strike'])
    opt_type = row.get('type', 'CALL')
    last = float(row.get('lastPrice', 0.0)) if not pd.isna(row.get('lastPrice', np.nan)) else 0.0
    iv = float(row.get('impliedVol', 0.0)) if not pd.isna(row.get('impliedVol', np.nan)) else 0.0
    oi = int(row.get('openInterest', 0)) if not pd.isna(row.get('openInterest', 0)) else 0
    vol = int(row.get('volume', 0)) if not pd.isna(row.get('volume', 0)) else 0
    exp = row.get('expiry_days', 0)

    moneyness = "ITM" if (underlying_price > strike and opt_type == "CALL") or (underlying_price < strike and opt_type == "PUT") else "OTM"
    near_atm = abs(underlying_price - strike) / underlying_price < 0.03

    reasons = []

    # Time decay
    if exp <= 7:
        reasons.append("short time → high theta decay")
    elif exp <= 30:
        reasons.append("near-term expiry → moderate theta")

    # IV signals
    if iv >= 0.6:
        reasons.append("high IV → premium elevated")
    elif iv <= 0.2:
        reasons.append("low IV → cheap premium")

    # Volatility classification
    reasons.append(f"volatility={vol_class}")

    # Trend + moneyness + volatility heuristics
    base = "MONITOR"
    if opt_type == "CALL":
        if trend == "bullish" and moneyness == "ITM":
            base = "BUY/HOLD (bullish ITM)"
        elif trend == "bullish" and near_atm:
            base = "BUY/HOLD (bullish near-ATM)"
        elif trend == "bearish" and moneyness == "ITM":
            base = "CONSIDER SELL (bearish trend)"
        elif vol_class == "high":
            base = "SELL premium (high vol)"
    else:  # PUT
        if trend == "bearish" and moneyness == "ITM":
            base = "BUY/HOLD (bearish ITM)"
        elif trend == "bearish" and near_atm:
            base = "BUY/HOLD (bearish near-ATM)"
        elif trend == "bullish" and moneyness == "ITM":
            base = "CONSIDER SELL (bullish trend)"
        elif vol_class == "high":
            base = "SELL premium (high vol)"

    reason_text = ", ".join(reasons) if reasons else "standard conditions"
    return f"{base} | {moneyness} | IV={iv:.2f} | OI={oi} | VOL={vol} | {reason_text}"

# --- Main routine ---
def run():
    tk = yf.Ticker(TICKER)

    try:
        price, trend, sma_short, sma_long, rsi, macd, macd_signal, vol = get_trend_and_price(tk)
        vol_class = classify_volatility(vol)
    except Exception as e:
        print("Failed to get price history:", e)
        sys.exit(1)

    print(f"\n{TICKER} ${price:.2f}")
    print(f"Trend: {trend} | SMA{SMA_SHORT}={sma_short:.2f} SMA{SMA_LONG}={sma_long:.2f} | RSI={rsi:.2f} | MACD={macd:.2f}/{macd_signal:.2f} | Vol={vol*100:.2f}%\n")

    expiries = tk.options
    if not expiries:
        print("No option expiries found for TSLA.")
        return

    expiries_to_check = expiries[:N_EXPIRIES]
    rows_out = []

    for exp in expiries_to_check:
        dte = days_to(exp)
        print(f"== Expiry: {exp} (DTE ~ {dte} days) ==")

        oc = tk.option_chain(exp)
        calls = oc.calls.copy()
        puts = oc.puts.copy()
        calls['expiry_days'] = dte
        puts['expiry_days'] = dte
        calls['type'] = 'CALL'
        puts['type'] = 'PUT'

        low = price * (1 - STRIKE_WINDOW_PCT)
        high = price * (1 + STRIKE_WINDOW_PCT)

        def select_frame(frame):
            in_window = frame[frame['strike'].between(low, high)]
            if in_window.empty:
                sel = frame.sort_values(by='openInterest', ascending=False).head(MAX_PER_SIDE)
            else:
                sel = in_window.sort_values(by='openInterest', ascending=False).head(MAX_PER_SIDE)
            return sel

        sel_calls = select_frame(calls)
        sel_puts = select_frame(puts)

        if not sel_calls.empty:
            print("\nCalls:")
            for _, r in sel_calls.iterrows():
                advice = advice_for_option(r, price, trend, vol_class)
                print(f" {int(r['strike']):>6}  last={r.get('lastPrice',0):>6}  IV={r.get('impliedVol',0):.2f}  OI={int(r.get('openInterest',0)):<6}  -> {advice}")
                rows_out.append({
                    "expiry": exp, "dte": dte, "type": "CALL", "strike": r['strike'],
                    "last": r.get('lastPrice', 0), "iv": r.get('impliedVol', 0),
                    "oi": r.get('openInterest', 0), "volume": r.get('volume', 0),
                    "advice": advice
                })

        if not sel_puts.empty:
            print("\nPuts:")
            for _, r in sel_puts.iterrows():
                advice = advice_for_option(r, price, trend, vol_class)
                print(f" {int(r['strike']):>6}  last={r.get('lastPrice',0):>6}  IV={r.get('impliedVol',0):.2f}  OI={int(r.get('openInterest',0)):<6}  -> {advice}")
                rows_out.append({
                    "expiry": exp, "dte": dte, "type": "PUT", "strike": r['strike'],
                    "last": r.get('lastPrice', 0), "iv": r.get('impliedVol', 0),
                    "oi": r.get('openInterest', 0), "volume": r.get('volume', 0),
                    "advice": advice
                })

        print("\n" + "-"*60 + "\n")

    if rows_out:
        df_out = pd.DataFrame(rows_out)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved advice to {OUTPUT_CSV} (rows: {len(rows_out)})")
    else:
        print("No option rows collected; no CSV written.")

if __name__ == "__main__":
    run()

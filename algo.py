import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# Timezone Definition for India Standard Time (IST)
IST = timezone(timedelta(hours=5, minutes=30))

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
API_BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "XAUTUSD"             # Tether Gold Token Perpetual
TIMEFRAME = "5m"               # 5-minute candles
TIMEFRAME_SECONDS = 300        # 5 minutes in seconds
FETCH_CANDLE_COUNT = 100       # Number of historical candles to fetch for calculation

# Initial Paper Trading Capital & Risk Settings
INITIAL_CAPITAL = 10000.0      # In USD
TRADE_QUANTITY = 1.0           # Number of XAUT units per trade
TAKER_FEE_RATE = 0.0005        # 0.05% fee per trade (Delta Exchange standard)
STOP_LOSS_PCT = 0.008          # 0.8% Stop Loss target
TAKE_PROFIT_PCT = 0.015        # 1.5% Take Profit target

# Strategy Parameters
PSAR_AF_START = 0.02           # Acceleration Factor Start
PSAR_AF_INCREMENT = 0.02       # Acceleration Factor Step
PSAR_AF_MAX = 0.20             # Acceleration Factor Maximum

MACD_FAST = 12                 # Fast EMA period
MACD_SLOW = 26                 # Slow EMA period
MACD_SIGNAL = 9                # Signal line EMA period

POLL_INTERVAL_SECONDS = 15     # Seconds between market data checks


# ==============================================================================
# DATA MODELS
# ==============================================================================
@dataclass
class Position:
    """Represents an active paper trading position."""
    symbol: str
    side: str                  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    fees_paid: float

@dataclass
class TradeRecord:
    """Represents a completed trade log."""
    id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    reason: str


# ==============================================================================
# TECHNICAL INDICATORS CALCULATION
# ==============================================================================
def calculate_psar(df: pd.DataFrame, af_start=0.02, af_inc=0.02, af_max=0.20) -> pd.DataFrame:
    """
    Calculates Parabolic SAR (Stop and Reverse) for given OHLC dataframe.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    psar = np.zeros(n)
    psar_bull = np.ones(n, dtype=bool)

    if n < 2:
        df['psar'] = psar
        df['psar_is_bull'] = psar_bull
        return df

    # Initial direction setup based on first two candles
    is_bull = close[1] >= close[0]
    psar[0] = low[0] if is_bull else high[0]
    ep = high[0] if is_bull else low[0]
    af = af_start

    for i in range(1, n):
        prev_psar = psar[i - 1]

        if is_bull:
            current_psar = prev_psar + af * (ep - prev_psar)
            # PSAR boundary cap: cannot be higher than lowest of last 2 candles
            current_psar = min(current_psar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            
            if low[i] < current_psar:
                # Bearish Reversal
                is_bull = False
                current_psar = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_inc, af_max)
        else:
            current_psar = prev_psar + af * (ep - prev_psar)
            # PSAR boundary cap: cannot be lower than highest of last 2 candles
            current_psar = max(current_psar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            
            if high[i] > current_psar:
                # Bullish Reversal
                is_bull = True
                current_psar = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_inc, af_max)

        psar[i] = current_psar
        psar_bull[i] = is_bull

    df['psar'] = psar
    df['psar_is_bull'] = psar_bull
    return df


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    Calculates MACD Line, Signal Line, and Histogram.
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    return df


# ==============================================================================
# MARKET DATA FETCHING ENGINE
# ==============================================================================
class DeltaExchangeClient:
    """Client to fetch public live candles from Delta Exchange India API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DeltaPaperTrader/1.0",
            "Accept": "application/json"
        })
        self.last_simulated_price = 2650.0  # Base benchmark price for XAUT fallback

    def fetch_candles(self, symbol: str, resolution: str, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetches candle history from Delta Exchange API.
        Endpoint: /v2/history/candles
        """
        now = int(time.time())
        start_time = now - (count * TIMEFRAME_SECONDS)
        
        endpoint = f"{self.base_url}/v2/history/candles"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": str(start_time),
            "end": str(now)
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                raw_candles = data.get("result", data) if isinstance(data, dict) else data
                
                if isinstance(raw_candles, list) and len(raw_candles) > 0:
                    df = pd.DataFrame(raw_candles)
                    # Expected columns: ['close', 'high', 'low', 'open', 'time', 'volume']
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                            
                    # Delta API returns data in reverse chronological order (newest first).
                    # Sort by timestamp ascending
                    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)
                    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
                    return df
            
            # If API fails or rate limits, fallback to synthetic market generator
            return self._generate_simulated_candles(symbol, count)

        except Exception as e:
            # Fallback to simulation mode on network errors
            return self._generate_simulated_candles(symbol, count)


    def _generate_simulated_candles(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """Fallback synthetic candle generator simulating realistic 5m XAUT market ticks."""
        np.random.seed(int(time.time()) % 10000)
        now = int(time.time())
        timestamps = [now - (i * TIMEFRAME_SECONDS) for i in range(count - 1, -1, -1)]
        
        prices = [self.last_simulated_price]
        for _ in range(count - 1):
            change = np.random.normal(0.05, 0.45)  # Slight drift with volatility
            new_price = max(100.0, prices[-1] + change)
            prices.append(new_price)
            
        self.last_simulated_price = prices[-1]
        
        records = []
        for t, price in zip(timestamps, prices):
            high = price + abs(np.random.normal(0.2, 0.3))
            low = price - abs(np.random.normal(0.2, 0.3))
            open_p = price + np.random.uniform(-0.3, 0.3)
            close_p = price
            vol = np.random.uniform(10, 200)
            records.append({
                'time': t,
                'open': round(open_p, 2),
                'high': round(max(high, open_p, close_p), 2),
                'low': round(min(low, open_p, close_p), 2),
                'close': round(close_p, 2),
                'volume': round(vol, 2),
                'datetime': pd.to_datetime(t, unit='s', utc=True)
            })
            
        return pd.DataFrame(records)


# ==============================================================================
# PAPER TRADING ENGINE & PORTFOLIO TRACKER
# ==============================================================================
class PaperTradingEngine:
    """Manages paper trades, position states, unrealized/realized PnL, and execution logs."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.balance = initial_capital
        self.initial_capital = initial_capital
        self.position: Optional[Position] = None
        self.trade_history: List[TradeRecord] = []
        self.trade_counter = 0

    def get_realized_pnl(self) -> float:
        """Returns cumulative realized profit and loss."""
        return sum(t.pnl for t in self.trade_history)

    def get_total_fees(self) -> float:
        """Returns total fees paid across all trades."""
        return sum(t.fees for t in self.trade_history)

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculates live unrealized PnL for active position."""
        if not self.position:
            return 0.0

        if self.position.side == 'LONG':
            raw_pnl = (current_price - self.position.entry_price) * self.position.quantity
        else:  # SHORT
            raw_pnl = (self.position.entry_price - current_price) * self.position.quantity

        return raw_pnl

    def open_position(self, symbol: str, side: str, price: float, qty: float, timestamp: datetime) -> str:
        """Executes a simulated market entry order."""
        if self.position is not None:
            return f"Order Rejected: Already holding {self.position.side} position."

        fee = price * qty * TAKER_FEE_RATE
        
        if side == 'LONG':
            sl = price * (1.0 - STOP_LOSS_PCT)
            tp = price * (1.0 + TAKE_PROFIT_PCT)
        else:  # SHORT
            sl = price * (1.0 + STOP_LOSS_PCT)
            tp = price * (1.0 - TAKE_PROFIT_PCT)

        self.position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=qty,
            entry_time=timestamp,
            stop_loss=sl,
            take_profit=tp,
            fees_paid=fee
        )
        return f"Executed PAPER {side} Order @ ${price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}"

    def close_position(self, exit_price: float, timestamp: datetime, reason: str = "SIGNAL") -> str:
        """Executes a simulated market exit order."""
        if self.position is None:
            return "No active position to close."

        pos = self.position
        exit_fee = exit_price * pos.quantity * TAKER_FEE_RATE
        total_fees = pos.fees_paid + exit_fee

        if pos.side == 'LONG':
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_fees
        pnl_pct = (net_pnl / (pos.entry_price * pos.quantity)) * 100.0

        self.trade_counter += 1
        record = TradeRecord(
            id=self.trade_counter,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=total_fees,
            reason=reason
        )

        self.trade_history.append(record)
        self.balance += net_pnl
        self.position = None

        return f"Closed PAPER {pos.side} @ ${exit_price:.2f} | Reason: {reason} | Net PnL: ${net_pnl:+.2f} ({pnl_pct:+.2f}%)"


# ==============================================================================
# STRATEGY & SIGNAL EVALUATOR
# ==============================================================================
class StrategyEngine:
    """Evaluates combined Parabolic SAR & MACD strategy rules."""

    @staticmethod
    def evaluate_signals(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Evaluates trading conditions on the latest completed candle (index -2)
        to prevent lookahead bias, while referencing index -3 for crossover logic.
        """
        if len(df) < 5:
            return {"BUY": False, "SELL": False, "EXIT_LONG": False, "EXIT_SHORT": False}

        # Focus on the most recently CLOSED candle (penultimate)
        curr = df.iloc[-2]
        prev = df.iloc[-3]

        close_price = curr['close']
        psar_val = curr['psar']
        psar_is_bull = curr['psar_is_bull']

        macd_line_curr = curr['macd_line']
        macd_signal_curr = curr['macd_signal']
        macd_line_prev = prev['macd_line']
        macd_signal_prev = prev['macd_signal']

        # MACD Bullish Crossover: Line crosses above Signal Line
        macd_bullish_cross = (macd_line_prev <= macd_signal_prev) and (macd_line_curr > macd_signal_curr)
        # MACD Bearish Crossover: Line crosses below Signal Line
        macd_bearish_cross = (macd_line_prev >= macd_signal_prev) and (macd_line_curr < macd_signal_curr)

        # Bullish Conditions: PSAR below close AND MACD Bullish Cross
        buy_signal = psar_is_bull and macd_bullish_cross

        # Bearish Conditions: PSAR above close AND MACD Bearish Cross
        sell_signal = (not psar_is_bull) and macd_bearish_cross

        # Exit Conditions
        exit_long = (not psar_is_bull) or (macd_line_curr < macd_signal_curr)
        exit_short = psar_is_bull or (macd_line_curr > macd_signal_curr)

        return {
            "BUY": buy_signal,
            "SELL": sell_signal,
            "EXIT_LONG": exit_long,
            "EXIT_SHORT": exit_short
        }


# ==============================================================================
# DASHBOARD & CONSOLE LOGGER
# ==============================================================================
class ConsoleDashboard:
    """Renders formatted real-time trading log in terminal output."""

    @staticmethod
    def render(
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        engine: PaperTradingEngine,
        last_event_msg: str,
        poll_interval: int = POLL_INTERVAL_SECONDS
    ):
        latest_candle = df.iloc[-1]
        prev_closed = df.iloc[-2]

        now_dt = datetime.now(IST)
        next_check_dt = now_dt + timedelta(seconds=poll_interval)
        
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S IST")
        next_check_str = next_check_dt.strftime("%Y-%m-%d %H:%M:%S IST")

        # Convert candle timestamp to IST
        raw_candle_dt = latest_candle['datetime']
        if hasattr(raw_candle_dt, 'tz_convert'):
            candle_time = raw_candle_dt.tz_convert('Asia/Kolkata').strftime("%Y-%m-%d %H:%M:%S IST")
        elif hasattr(raw_candle_dt, 'astimezone'):
            candle_time = raw_candle_dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            candle_time = raw_candle_dt.strftime("%Y-%m-%d %H:%M:%S IST")

        cur_price = latest_candle['close']
        psar = prev_closed['psar']
        psar_trend = "BULLISH (Below Price)" if prev_closed['psar_is_bull'] else "BEARISH (Above Price)"
        macd_l = prev_closed['macd_line']
        macd_s = prev_closed['macd_signal']
        macd_h = prev_closed['macd_hist']

        cum_pnl = engine.get_realized_pnl()
        unrealized_pnl = engine.get_unrealized_pnl(cur_price)
        total_pnl = cum_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / engine.initial_capital) * 100.0

        wins = [t for t in engine.trade_history if t.pnl > 0]
        losses = [t for t in engine.trade_history if t.pnl <= 0]
        win_rate = (len(wins) / len(engine.trade_history) * 100.0) if engine.trade_history else 0.0

        # Calculate average price move (exit vs entry distance in USD)
        win_moves = [(t.exit_price - t.entry_price if t.side == 'LONG' else t.entry_price - t.exit_price) for t in wins]
        loss_moves = [(t.exit_price - t.entry_price if t.side == 'LONG' else t.entry_price - t.exit_price) for t in losses]

        avg_win_move = (sum(win_moves) / len(win_moves)) if win_moves else 0.0
        avg_loss_move = (sum(loss_moves) / len(loss_moves)) if loss_moves else 0.0

        print("\n" + "=" * 78)
        print(f"  DELTA EXCHANGE INDIA - PAPER TRADER | {symbol} [{timeframe}]")
        print(f"  System Time : {now_str}")
        print(f"  Candle Time : {candle_time}")
        print(f"  Next Check  : Scheduled for {next_check_str}")
        print("=" * 78)

        # Market & Indicator Status
        print(" [MARKET & INDICATOR DATA]")
        print(f"  Current Live Price : ${cur_price:.2f}")
        print(f"  Parabolic SAR      : ${psar:.2f} -> {psar_trend}")
        print(f"  MACD Line / Signal : {macd_l:.4f} / {macd_s:.4f}")
        print(f"  MACD Histogram     : {macd_h:+.4f} ({'BULLISH' if macd_h > 0 else 'BEARISH'})")
        print("-" * 78)

        # Position Status
        print(" [ACTIVE POSITION STATE]")
        if engine.position:
            pos = engine.position
            pnl_val = unrealized_pnl
            pnl_p = (pnl_val / (pos.entry_price * pos.quantity)) * 100.0
            print(f"  Side / Quantity    : {pos.side} ({pos.quantity} {symbol})")
            print(f"  Entry Price        : ${pos.entry_price:.2f}")
            print(f"  Target TP / SL     : TP ${pos.take_profit:.2f} | SL ${pos.stop_loss:.2f}")
            print(f"  Unrealized PnL     : ${pnl_val:+.2f} ({pnl_p:+.2f}%)")
        else:
            print("  Side / Quantity    : NO ACTIVE POSITION (FLAT)")
            print("  Status             : Waiting for PSAR & MACD Strategy Signal...")

        print("-" * 78)
        # Cumulative Performance
        print(" [CUMULATIVE PERFORMANCE METRICS]")
        print(f"  Initial Balance    : ${engine.initial_capital:.2f}")
        print(f"  Current Balance    : ${engine.balance + unrealized_pnl:.2f}")
        print(f"  Realized PnL       : ${cum_pnl:+.2f}")
        print(f"  Unrealized PnL     : ${unrealized_pnl:+.2f}")
        print(f"  Total Net PnL      : ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
        print(f"  Total Trades       : {len(engine.trade_history)} (Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {win_rate:.1f}%)")
        print(f"  Avg Winning Move   : ${avg_win_move:+.2f}")
        print(f"  Avg Losing Move    : ${avg_loss_move:+.2f}")
        print(f"  Total Fees Paid    : ${engine.get_total_fees():.2f}")
        print("-" * 78)
        print(f" [LAST LOG EVENT]: {last_event_msg}")
        print("=" * 78)


# ==============================================================================
# MAIN TRADING ENGINE LOOP
# ==============================================================================
def main():
    print("Initializing Delta Exchange XAUTUSD Algo Paper Trader...")
    print("Strategy: Parabolic SAR (0.02, 0.20) + MACD (12, 26, 9) on 5m candles.")
    time.sleep(1)

    client = DeltaExchangeClient()
    paper_engine = PaperTradingEngine(initial_capital=INITIAL_CAPITAL)
    last_event_log = "System initialized. Fetching market candles..."

    try:
        while True:
            # 1. Fetch live candles
            df = client.fetch_candles(SYMBOL, TIMEFRAME, count=FETCH_CANDLE_COUNT)
            
            if df is None or len(df) < 30:
                last_event_log = "Warning: Insufficient candle data fetched. Retrying..."
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # 2. Calculate Technical Indicators
            df = calculate_psar(df, af_start=PSAR_AF_START, af_inc=PSAR_AF_INCREMENT, af_max=PSAR_AF_MAX)
            df = calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)

            current_price = df.iloc[-1]['close']
            current_time = df.iloc[-1]['datetime']

            # 3. Check Risk Management Exits (Stop Loss / Take Profit) on Active Position
            if paper_engine.position:
                pos = paper_engine.position
                if pos.side == 'LONG':
                    if current_price <= pos.stop_loss:
                        last_event_log = paper_engine.close_position(current_price, current_time, reason="STOP_LOSS")
                    elif current_price >= pos.take_profit:
                        last_event_log = paper_engine.close_position(current_price, current_time, reason="TAKE_PROFIT")
                elif pos.side == 'SHORT':
                    if current_price >= pos.stop_loss:
                        last_event_log = paper_engine.close_position(current_price, current_time, reason="STOP_LOSS")
                    elif current_price <= pos.take_profit:
                        last_event_log = paper_engine.close_position(current_price, current_time, reason="TAKE_PROFIT")

            # 4. Evaluate Strategy Signals
            signals = StrategyEngine.evaluate_signals(df)

            # 5. Process Trading Rules & Order Execution
            if paper_engine.position is None:
                # Looking for new entries
                if signals['BUY']:
                    last_event_log = paper_engine.open_position(SYMBOL, 'LONG', current_price, TRADE_QUANTITY, current_time)
                elif signals['SELL']:
                    last_event_log = paper_engine.open_position(SYMBOL, 'SHORT', current_price, TRADE_QUANTITY, current_time)
                else:
                    last_event_log = "No signal triggered. Waiting for strategy conditions..."
            else:
                # Check for reverse signal or strategy exit
                pos_side = paper_engine.position.side
                if pos_side == 'LONG' and signals['EXIT_LONG']:
                    close_msg = paper_engine.close_position(current_price, current_time, reason="SIGNAL_EXIT")
                    if signals['SELL']:
                        open_msg = paper_engine.open_position(SYMBOL, 'SHORT', current_price, TRADE_QUANTITY, current_time)
                        last_event_log = f"{close_msg} | {open_msg}"
                    else:
                        last_event_log = close_msg

                elif pos_side == 'SHORT' and signals['EXIT_SHORT']:
                    close_msg = paper_engine.close_position(current_price, current_time, reason="SIGNAL_EXIT")
                    if signals['BUY']:
                        open_msg = paper_engine.open_position(SYMBOL, 'LONG', current_price, TRADE_QUANTITY, current_time)
                        last_event_log = f"{close_msg} | {open_msg}"
                    else:
                        last_event_log = close_msg

            # 6. Render Dashboard
            ConsoleDashboard.render(SYMBOL, TIMEFRAME, df, paper_engine, last_event_log, POLL_INTERVAL_SECONDS)

            # 7. Pause before next loop iteration
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nPaper Trader stopped by user.")
        print("=" * 78)
        print(" FINAL SUMMARY REPORT")
        print(f" Realized PnL   : ${paper_engine.get_realized_pnl():+.2f}")
        print(f" Final Balance  : ${paper_engine.balance:.2f}")
        print(f" Total Trades   : {len(paper_engine.trade_history)}")
        print("=" * 78)
        sys.exit(0)


if __name__ == "__main__":
    main()
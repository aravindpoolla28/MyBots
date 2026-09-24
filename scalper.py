#!/usr/bin/env python3
import json
import os
import sys
import time
import threading
from datetime import datetime
from collections import deque
import pandas as pd
import requests
import websocket
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONFIGURATION PARAMETERS & TELEGRAM ENV
# ==========================================
BASE_URL = "https://api.india.delta.exchange"         # Delta Exchange India REST Base
WS_URL = "wss://public-socket.india.delta.exchange"    # Delta Exchange India Public WS Endpoint

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOL = "BTCUSD"         # Target trading pair
PAPER_BALANCE = 10000.0   # Starting Virtual Balance ($)
TRADE_SIZE_USD = 1000.0   # Target position trade size ($)

# Strategy Settings
TIMEFRAME_MINUTES = 5     # Strategy bar timeframe
SWING_LOOKBACK_BARS = 10  # Lookback for swing highs/lows

# Sensitivity Metrics
ABSORPTION_DELTA_THRESHOLD = 750.0  # Net Volume Delta threshold
IMBALANCE_RATIO = 4.0             # 4:1 orderbook imbalance threshold
SMOOTHING_WINDOW_SEC = 5          # Rolling window for orderbook depth smoothing

LOG_INTERVAL_SECONDS = 5  # Terminal refresh rate
TOUCH_COOLDOWN_SEC = 60   # Seconds to pause Telegram touch alerts to avoid spamming

# Tracking last alert timestamps to prevent spam
last_swing_low_alert_time = 0.0
last_swing_high_alert_time = 0.0

# ==========================================
# TELEGRAM NOTIFICATION HELPER
# ==========================================
def send_telegram_notification(message: str):
    """ Sends a Telegram notification asynchronously in a non-blocking thread. """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"\n[Telegram Error] Failed to send message: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ==========================================
# REAL-TIME ORDERBOOK & TICK ENGINE
# ==========================================
class OrderFlowEngine:
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = {}
        self.asks = {}
        self.trades = []
        self.imbalance_history = deque()
        self.current_price = 0.0
        self.lock = threading.Lock()
        self.ws = None

    def start(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                with self.lock:
                    if msg_type == "trades":
                        price = float(data.get("p", 0.0))
                        size = float(data.get("s", 0.0))
                        role = data.get("r", "t")
                        
                        if price > 0:
                            self.current_price = price
                            
                        if size > 0:
                            side = "buy" if role == "m" else "sell"
                            self.trades.append({
                                "price": price,
                                "size": size,
                                "side": side,
                                "timestamp": time.time()
                            })

                    elif msg_type == "ob_l2":
                        if "a" in data and isinstance(data["a"], list):
                            self.asks = {float(item[0]): float(item[1]) for item in data["a"] if float(item[1]) > 0}
                        if "b" in data and isinstance(data["b"], list):
                            self.bids = {float(item[0]): float(item[1]) for item in data["b"] if float(item[1]) > 0}
                        
                        self._update_imbalance_queue()

                    elif msg_type == "ticker":
                        price = float(data.get("close") or data.get("mark_price") or data.get("p") or 0.0)
                        if price > 0:
                            self.current_price = price

            except Exception:
                pass

        def on_open(ws):
            sub_msg = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {"name": "trades", "symbols": [self.symbol]},
                        {"name": "ob_l2", "symbols": [self.symbol]},
                        {"name": "ticker", "symbols": [self.symbol]}
                    ]
                }
            }
            ws.send(json.dumps(sub_msg))

        def run_ws():
            self.ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=lambda ws, err: None,
                on_close=lambda ws, code, msg: None
            )
            self.ws.run_forever()

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

    def _update_imbalance_queue(self, depth=5):
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        
        total_bid_vol = sum([size for _, size in sorted_bids])
        total_ask_vol = sum([size for _, size in sorted_asks])
        
        raw_imb = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
        now = time.time()
        
        self.imbalance_history.append((now, raw_imb))
        
        while self.imbalance_history and self.imbalance_history[0][0] < now - SMOOTHING_WINDOW_SEC:
            self.imbalance_history.popleft()

    def get_smoothed_imbalance(self):
        with self.lock:
            if not self.imbalance_history:
                return 1.0
            total_imb = sum([imb for _, imb in self.imbalance_history])
            return total_imb / len(self.imbalance_history)

    def calculate_cvd(self, time_window_sec=300):
        now = time.time()
        with self.lock:
            self.trades = [t for t in self.trades if now - t["timestamp"] <= time_window_sec]
            buy_vol = sum([t["size"] for t in self.trades if t["side"] == "buy"])
            sell_vol = sum([t["size"] for t in self.trades if t["side"] == "sell"])
            return buy_vol - sell_vol, buy_vol, sell_vol

# ==========================================
# REST API FETCHING & FALLBACKS
# ==========================================
def fetch_ticker_price(symbol):
    url = f"{BASE_URL}/v2/tickers/{symbol}"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            result = res.json().get("result", {})
            return float(result.get("close") or result.get("mark_price") or 0.0)
    except Exception:
        pass
    return 0.0

def fetch_candles(symbol, resolution="5m", limit=30):
    url = f"{BASE_URL}/v2/history/candles"
    now_ts = int(time.time())
    start_ts = now_ts - (limit * 5 * 60)
    
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": start_ts,
        "end": now_ts
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("success"):
                result = data.get("result", [])
                if isinstance(result, list) and len(result) > 0:
                    df = pd.DataFrame(result)
                    for col in ["close", "high", "low", "open"]:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                    return df
    except Exception:
        pass
    return pd.DataFrame()

# ==========================================
# PAPER TRADER ENGINE
# ==========================================
class PaperTrader:
    def __init__(self, balance):
        self.balance = balance
        self.position = None
        self.last_log = "Engine started. Scanning market flow..."

    def execute_signal(self, signal, current_price, stop_loss, reason_str, cvd, imbalance, swing_low, swing_high):
        if self.position is not None:
            return

        self.position = {
            "side": signal,
            "entry": current_price,
            "size_usd": TRADE_SIZE_USD,
            "stop": stop_loss,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        self.last_log = f"🚀 EXECUTED {signal} @ ${current_price:.2f} | Stop:${stop_loss:.2f}"

        tg_msg = (
            f"🚀 *ORDER FLOW TRADE EXECUTED*\n\n"
            f"*Symbol:* {SYMBOL}\n"
            f"*Action:* {signal}\n"
            f"*Reason:* {reason_str}\n"
            f"*Entry Price:* ${current_price:.2f}\n"
            f"*Stop Loss:* ${stop_loss:.2f}\n"
            f"*Position Size:* ${TRADE_SIZE_USD:,.2f}\n\n"
            f"📊 *Metrics:* CVD: `{cvd:+.2f}` | Imbalance: `{imbalance:.2f}x`\n"
            f"📍 *Swings:* Low `${swing_low:.2f}` | High `${swing_high:.2f}`"
        )
        send_telegram_notification(tg_msg)

    def manage_positions(self, current_price):
        if self.position is None or current_price == 0.0:
            return

        side = self.position["side"]
        entry = self.position["entry"]
        stop = self.position["stop"]
        size_usd = self.position["size_usd"]

        pct_change = (current_price - entry) / entry if side == "BUY" else (entry - current_price) / entry
        pnl = size_usd * pct_change

        # Stop Loss Trigger
        if (side == "BUY" and current_price <= stop) or (side == "SELL" and current_price >= stop):
            self.balance += pnl
            self.last_log = f"❌ STOP LOSS HIT @ ${current_price:.2f} \\vert{{}} PnL:${pnl:+.2f}"
            
            tg_msg = (
                f"🛑 *POSITION CLOSED - STOP LOSS*\n\n"
                f"*Symbol:* {SYMBOL}\n"
                f"*Side:* {side}\n"
                f"*Entry:* ${entry:.2f} \\vert{{}} *Exit:*${current_price:.2f}\n"
                f"*Realized PnL:* `${pnl:+.2f}`\n"
                f"*New Balance:* `${self.balance:,.2f}`"
            )
            send_telegram_notification(tg_msg)
            self.position = None

        # Take Profit Trigger (1.5x R:R)
        else:
            risk = abs(entry - stop)
            target = entry + (1.5 * risk) if side == "BUY" else entry - (1.5 * risk)
            if (side == "BUY" and current_price >= target) or (side == "SELL" and current_price <= target):
                self.balance += pnl
                self.last_log = f"🎯 TAKE PROFIT HIT @ ${current_price:.2f} \\vert{{}} PnL:${pnl:+.2f}"
                
                tg_msg = (
                    f"🎯 *POSITION CLOSED - TAKE PROFIT*\n\n"
                    f"*Symbol:* {SYMBOL}\n"
                    f"*Side:* {side}\n"
                    f"*Entry:* ${entry:.2f} \\vert{{}} *Exit:*${current_price:.2f}\n"
                    f"*Realized PnL:* `${pnl:+.2f}`\n"
                    f"*New Balance:* `${self.balance:,.2f}`"
                )
                send_telegram_notification(tg_msg)
                self.position = None

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    global last_swing_low_alert_time, last_swing_high_alert_time

    print("================================================================================")
    print(f" 🟢 DELTA EXCHANGE ORDER FLOW ENGINE | SYMBOL: {SYMBOL}")
    print(f" ⚙️ SENSITIVITY CONFIG: CVD Delta Threshold = ±{ABSORPTION_DELTA_THRESHOLD} | Imb Ratio = {IMBALANCE_RATIO}x")
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(" 📲 TELEGRAM NOTIFICATIONS: ENABLED")
        send_telegram_notification(f"🟢 *Order Flow Bot Online*\nMonitoring `{SYMBOL}` on Delta Exchange India.")
    else:
        print(" ⚠️ TELEGRAM NOTIFICATIONS: DISABLED (Missing env variables TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
    print("================================================================================")

    engine = OrderFlowEngine(SYMBOL)
    engine.start()
    trader = PaperTrader(PAPER_BALANCE)

    time.sleep(2)
    
    next_check_time = 0.0
    swing_high = 0.0
    swing_low = 0.0

    while True:
        now = time.time()
        
        # 1. Periodically update swing levels
        if now >= next_check_time or swing_high == 0.0:
            candles = fetch_candles(SYMBOL, resolution=f"{TIMEFRAME_MINUTES}m", limit=SWING_LOOKBACK_BARS + 2)
            if not candles.empty:
                swing_high = candles["high"].iloc[:-1].max()
                swing_low = candles["low"].iloc[:-1].min()

            next_check_time = now + (TIMEFRAME_MINUTES * 60)

        # 2. Get real-time price & flow indicators
        current_price = engine.current_price
        if current_price == 0.0:
            current_price = fetch_ticker_price(SYMBOL)

        cvd, buy_vol, sell_vol = engine.calculate_cvd(time_window_sec=TIMEFRAME_MINUTES * 60)
        imbalance = engine.get_smoothed_imbalance()

        # 3. Evaluate Signals & Generate Detailed Swing Audit Logs
        if current_price > 0.0:
            trader.manage_positions(current_price)

            signal = None
            stop_loss = 0.0
            reason_str = ""

            # Check if price is interacting with Swing Low (within 0.05% or below)
            if swing_low > 0.0 and current_price <= (swing_low * 1.0005):
                # Condition evaluations for Swing Low
                c_price_low = current_price <= swing_low
                c_cvd_abs_low = cvd < -ABSORPTION_DELTA_THRESHOLD
                c_imb_abs_low = imbalance > IMBALANCE_RATIO
                c_cvd_bk_low = cvd < -ABSORPTION_DELTA_THRESHOLD
                c_imb_bk_low = imbalance < (1 / IMBALANCE_RATIO)

                # 1. Reversal Long Check
                if c_price_low and c_cvd_abs_low and c_imb_abs_low:
                    signal = "BUY"
                    stop_loss = current_price * 0.997
                    reason_str = "Seller Absorption Reversal (LONG)"
                # 2. Breakdown Short Check
                elif c_price_low and c_cvd_bk_low and c_imb_bk_low:
                    signal = "SELL"
                    stop_loss = swing_low * 1.002
                    reason_str = "Aggressive Breakdown Continuation (SHORT)"

                # Telegram Diagnostic Audit Dispatcher for Swing Low Touch
                if (now - last_swing_low_alert_time) > TOUCH_COOLDOWN_SEC:
                    trade_status = f"✅ *PAPER TRADE EXECUTED ({signal})*" if signal else "❌ *NO TRADE EXECUTED*"
                    
                    audit_msg = (
                        f"📍 *SWING LOW TOUCHED AUDIT*\n\n"
                        f"*Symbol:* `{SYMBOL}`\n"
                        f"*Current Price:* `${current_price:.2f}`\n"
                        f"*Swing Low Level:* `${swing_low:.2f}`\n\n"
                        f"📊 *Market Flow Snapshot:*\n"
                        f"• CVD: `{cvd:+.2f}` (Target: `<-{ABSORPTION_DELTA_THRESHOLD:.0f}`)\n"
                        f"• Imbalance: `{imbalance:.2f}x` (Abs Target: `>{IMBALANCE_RATIO:.1f}x` | Bk Target: `<{1/IMBALANCE_RATIO:.2f}x`)\n\n"
                        f"📋 *Criteria Audit:*\n"
                        f"• Price <= Swing Low: {'✅ PASS' if c_price_low else '❌ FAIL'}\n"
                        f"• Reversal CVD (<-750): {'✅ PASS' if c_cvd_abs_low else '❌ FAIL'}\n"
                        f"• Reversal Imbalance (>4.0x): {'✅ PASS' if c_imb_abs_low else '❌ FAIL'}\n"
                        f"• Breakdown Imbalance (<0.25x): {'✅ PASS' if c_imb_bk_low else '❌ FAIL'}\n\n"
                        f"*Result:* {trade_status}"
                    )
                    send_telegram_notification(audit_msg)
                    last_swing_low_alert_time = now

            # Check if price is interacting with Swing High (within 0.05% or above)
            elif swing_high > 0.0 and current_price >= (swing_high * 0.9995):
                # Condition evaluations for Swing High
                c_price_high = current_price >= swing_high
                c_cvd_abs_high = cvd > ABSORPTION_DELTA_THRESHOLD
                c_imb_abs_high = imbalance < (1 / IMBALANCE_RATIO)
                c_cvd_bk_high = cvd > ABSORPTION_DELTA_THRESHOLD
                c_imb_bk_high = imbalance > IMBALANCE_RATIO

                # 1. Reversal Short Check
                if c_price_high and c_cvd_abs_high and c_imb_abs_high:
                    signal = "SELL"
                    stop_loss = current_price * 1.003
                    reason_str = "Buyer Absorption Reversal (SHORT)"
                # 2. Breakout Long Check
                elif c_price_high and c_cvd_bk_high and c_imb_bk_high:
                    signal = "BUY"
                    stop_loss = swing_high * 0.998
                    reason_str = "Aggressive Breakout Continuation (LONG)"

                # Telegram Diagnostic Audit Dispatcher for Swing High Touch
                if (now - last_swing_high_alert_time) > TOUCH_COOLDOWN_SEC:
                    trade_status = f"✅ *PAPER TRADE EXECUTED ({signal})*" if signal else "❌ *NO TRADE EXECUTED*"
                    
                    audit_msg = (
                        f"📍 *SWING HIGH TOUCHED AUDIT*\n\n"
                        f"*Symbol:* `{SYMBOL}`\n"
                        f"*Current Price:* `${current_price:.2f}`\n"
                        f"*Swing High Level:* `${swing_high:.2f}`\n\n"
                        f"📊 *Market Flow Snapshot:*\n"
                        f"• CVD: `{cvd:+.2f}` (Target: `>{ABSORPTION_DELTA_THRESHOLD:.0f}`)\n"
                        f"• Imbalance: `{imbalance:.2f}x` (Abs Target: `<{1/IMBALANCE_RATIO:.2f}x` | Bk Target: `>{IMBALANCE_RATIO:.1f}x`)\n\n"
                        f"📋 *Criteria Audit:*\n"
                        f"• Price >= Swing High: {'✅ PASS' if c_price_high else '❌ FAIL'}\n"
                        f"• CVD Threshold (>+750): {'✅ PASS' if c_cvd_abs_high else '❌ FAIL'}\n"
                        f"• Reversal Imbalance (<0.25x): {'✅ PASS' if c_imb_abs_high else '❌ FAIL'}\n"
                        f"• Breakout Imbalance (>4.0x): {'✅ PASS' if c_imb_bk_high else '❌ FAIL'}\n\n"
                        f"*Result:* {trade_status}"
                    )
                    send_telegram_notification(audit_msg)
                    last_swing_high_alert_time = now

            if signal:
                trader.execute_signal(
                    signal=signal,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    reason_str=reason_str,
                    cvd=cvd,
                    imbalance=imbalance,
                    swing_low=swing_low,
                    swing_high=swing_high
                )

        # 4. Print Continuous Terminal Log
        remaining_sec = int(max(0, next_check_time - time.time()))
        mins, secs = divmod(remaining_sec, 60)
        timestamp = datetime.now().strftime('%H:%M:%S')

        pos_str = "FLAT"
        if trader.position:
            pos_str = f"{trader.position['side']} @ ${trader.position['entry']:.2f}"

        dist_to_low = current_price - swing_low if swing_low > 0 else 0.0
        dist_to_high = swing_high - current_price if swing_high > 0 else 0.0

        print(f"[{timestamp}] Price: ${current_price:.2f} | Swings: [🔻 ${swing_low:.2f} ({dist_to_low:+.1f}p) \\vert{{}} 🔺 ${swing_high:.2f} ({dist_to_high:+.1f}p)] | CVD: {cvd:+.2f} | Imb: {imbalance:.2f} | Pos: {pos_str} | Status: {trader.last_log}")

        time.sleep(LOG_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Order Flow Engine...")
        sys.exit(0)
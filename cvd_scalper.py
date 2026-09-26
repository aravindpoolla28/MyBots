#!/usr/bin/env python3
"""
Binance Spot & Futures Cumulative Volume Delta (CVD) Scalping Monitor
--------------------------------------------------------------------
Monitors real-time trade flow from Binance Spot and USDS-M Futures (@aggTrade),
computes Cumulative Volume Delta (CVD), rolling 1m/5m deltas, basis spread,
delta z-scores, and identifies high-probability scalping setups (divergences,
spot-led breakouts, futures exhaustion, and absorption).
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from collections import deque
import statistics
import requests
import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# LOGGING SETUP
# ==========================================
logger = logging.getLogger("CVD_Scalper")
logger.setLevel(logging.INFO if os.environ.get("DEBUG") else logging.WARNING)
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
SYMBOL = "BTCUSDT"
SPOT_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
FUTURES_WS_URL = "wss://fstream.binancefuture.com/ws/btcusdt@aggTrade"
FUTURES_FALLBACK_URL = "wss://stream.binancefuture.com/ws/btcusdt@aggTrade"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Signal thresholds
Z_SCORE_THRESHOLD = 2.5       # Standard deviations for volume delta spike
DIVERGENCE_LOOKBACK_SEC = 300 # 5 minutes
MIN_DELTA_THRESHOLD = 5.0     # Minimum BTC delta for significant imbalance
REFRESH_INTERVAL_SEC = 2.0    # Terminal UI refresh rate

# ==========================================
# TELEGRAM NOTIFICATIONS
# ==========================================
def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


# ==========================================
# CVD & ORDER FLOW ENGINE
# ==========================================
class BinanceCVDMonitor:
    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol
        self.lock = threading.Lock()
        
        # Prices & Basis
        self.spot_price = 0.0
        self.futures_price = 0.0
        self.basis = 0.0
        self.basis_bps = 0.0

        # Trade buffers: list of dicts {"price": float, "qty": float, "side": "buy"/"sell", "timestamp": float}
        self.spot_trades = deque()
        self.futures_trades = deque()

        # Cumulative counters (since startup)
        self.spot_cvd_btc = 0.0
        self.futures_cvd_btc = 0.0
        self.spot_cvd_usd = 0.0
        self.futures_cvd_usd = 0.0

        # WebSocket instances
        self.spot_ws = None
        self.futures_ws = None
        self._stop = False

        # Active Signal tracking
        self.last_signal_time = 0.0
        self.current_signal = None

    def start(self):
        # Start REST initial price fetch
        self._fetch_initial_prices()

        # Start Spot WS
        t_spot = threading.Thread(target=self._run_spot_ws, daemon=True)
        t_spot.start()

        # Start Futures WS
        t_futures = threading.Thread(target=self._run_futures_ws, daemon=True)
        t_futures.start()

    def stop(self):
        self._stop = True
        try:
            if self.spot_ws: self.spot_ws.close()
            if self.futures_ws: self.futures_ws.close()
        except Exception:
            pass

    def _fetch_initial_prices(self):
        try:
            r1 = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3)
            if r1.status_code == 200:
                self.spot_price = float(r1.json().get("price", 0.0))
            r2 = requests.get("https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT", timeout=3)
            if r2.status_code == 200:
                self.futures_price = float(r2.json().get("price", 0.0))
        except Exception:
            pass

    def _run_spot_ws(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("e") == "aggTrade":
                    price = float(data["p"])
                    qty = float(data["q"])
                    is_buyer_maker = data["m"] # True = sell taker, False = buy taker
                    side = "sell" if is_buyer_maker else "buy"
                    now = time.time()

                    with self.lock:
                        self.spot_price = price
                        self.spot_trades.append({"price": price, "qty": qty, "side": side, "timestamp": now})
                        delta_btc = qty if side == "buy" else -qty
                        delta_usd = delta_btc * price
                        self.spot_cvd_btc += delta_btc
                        self.spot_cvd_usd += delta_usd
                        self._update_basis()
            except Exception as e:
                logger.debug(f"Spot WS parse error: {e}")

        def run():
            while not self._stop:
                try:
                    self.spot_ws = websocket.WebSocketApp(
                        SPOT_WS_URL,
                        on_message=on_message,
                        on_error=lambda ws, err: logger.debug(f"Spot WS error: {err}"),
                        on_close=lambda ws, c, m: logger.debug("Spot WS closed")
                    )
                    self.spot_ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    pass
                if not self._stop:
                    time.sleep(3)
        run()

    def _run_futures_ws(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("e") == "aggTrade":
                    price = float(data["p"])
                    qty = float(data["q"])
                    is_buyer_maker = data["m"]
                    side = "sell" if is_buyer_maker else "buy"
                    now = time.time()

                    with self.lock:
                        self.futures_price = price
                        self.futures_trades.append({"price": price, "qty": qty, "side": side, "timestamp": now})
                        delta_btc = qty if side == "buy" else -qty
                        delta_usd = delta_btc * price
                        self.futures_cvd_btc += delta_btc
                        self.futures_cvd_usd += delta_usd
                        self._update_basis()
            except Exception as e:
                logger.debug(f"Futures WS parse error: {e}")

        def run():
            urls = [FUTURES_WS_URL, FUTURES_FALLBACK_URL, "wss://fstream.binance.com/ws/btcusdt@aggTrade"]
            idx = 0
            while not self._stop:
                url = urls[idx % len(urls)]
                try:
                    self.futures_ws = websocket.WebSocketApp(
                        url,
                        on_message=on_message,
                        on_error=lambda ws, err: logger.debug(f"Futures WS error ({url}): {err}"),
                        on_close=lambda ws, c, m: logger.debug(f"Futures WS closed ({url})")
                    )
                    self.futures_ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    pass
                idx += 1
                if not self._stop:
                    time.sleep(3)
        run()

    def _update_basis(self):
        if self.spot_price > 0 and self.futures_price > 0:
            self.basis = self.futures_price - self.spot_price
            self.basis_bps = (self.basis / self.spot_price) * 10000.0

    def cleanup_old_trades(self, max_sec=900):
        now = time.time()
        with self.lock:
            while self.spot_trades and now - self.spot_trades[0]["timestamp"] > max_sec:
                self.spot_trades.popleft()
            while self.futures_trades and now - self.futures_trades[0]["timestamp"] > max_sec:
                self.futures_trades.popleft()

    def get_window_delta(self, trades_deque, window_sec):
        now = time.time()
        buy_sum = 0.0
        sell_sum = 0.0
        with self.lock:
            for t in trades_deque:
                if now - t["timestamp"] <= window_sec:
                    if t["side"] == "buy":
                        buy_sum += t["qty"]
                    else:
                        sell_sum += t["qty"]
        return buy_sum - sell_sum, buy_sum, sell_sum


# ==========================================
# SCALPING STRATEGY & SIGNAL GENERATOR
# ==========================================
class ScalpSignalEngine:
    def __init__(self, monitor: BinanceCVDMonitor):
        self.monitor = monitor
        # Keep history of recent 1m deltas to compute z-scores
        self.spot_1m_history = deque(maxlen=60)
        self.futures_1m_history = deque(maxlen=60)
        self.last_price_check = 0.0
        self.price_history = deque(maxlen=300) # 5 mins of prices

    def evaluate(self):
        self.monitor.cleanup_old_trades()

        # Get windows
        spot_1m, _, _ = self.monitor.get_window_delta(self.monitor.spot_trades, 60)
        fut_1m, _, _ = self.monitor.get_window_delta(self.monitor.futures_trades, 60)
        spot_5m, _, _ = self.monitor.get_window_delta(self.monitor.spot_trades, 300)
        fut_5m, _, _ = self.monitor.get_window_delta(self.monitor.futures_trades, 300)

        spot_p = self.monitor.spot_price
        fut_p = self.monitor.futures_price
        basis = self.monitor.basis
        basis_bps = self.monitor.basis_bps

        if spot_p == 0 or fut_p == 0:
            return None

        # Track price history for swing/trend analysis
        now = time.time()
        if now - self.last_price_check >= 5.0:
            self.last_price_check = now
            self.price_history.append((now, spot_p))
            self.spot_1m_history.append(spot_1m)
            self.futures_1m_history.append(fut_1m)

        # Compute z-score for futures 1m delta if enough history
        fut_z = 0.0
        if len(self.futures_1m_history) >= 10:
            mean_d = statistics.mean(self.futures_1m_history)
            stdev_d = statistics.pstdev(self.futures_1m_history)
            if stdev_d > 0:
                fut_z = (fut_1m - mean_d) / stdev_d

        # Determine price trend over last 3-5 mins
        price_trend = "FLAT"
        if len(self.price_history) >= 10:
            old_p = self.price_history[0][1]
            if spot_p > old_p * 1.0003:
                price_trend = "UP"
            elif spot_p < old_p * 0.9997:
                price_trend = "DOWN"

        # Signal rules & Rationale
        signal = None
        setup_type = ""
        confidence = 5
        rationale = ""
        sl = 0.0
        tp1 = 0.0
        tp2 = 0.0

        # RULE 1: Spot-Led Strong Rally (High Confidence Long)
        if spot_5m > 30 and fut_5m > 0 and spot_1m > 5 and price_trend != "DOWN":
            setup_type = "SPOT-LED MOMENTUM LONG"
            signal = "LONG"
            confidence = 8
            sl = spot_p - 120.0
            tp1 = spot_p + 150.0
            tp2 = spot_p + 300.0
            rationale = (f"Strong organic spot buying (5m Spot CVD: +{spot_5m:.1f} BTC) "
                         f"leading price higher. Healthy continuation scalp.")

        # RULE 2: Futures FOMO Spike / Divergence (Short Scalp / Squeeze Warning)
        elif fut_5m > 150 and spot_5m < 10 and fut_z > Z_SCORE_THRESHOLD:
            setup_type = "FUTURES FOMO EXHAUSTION SHORT"
            signal = "SHORT"
            confidence = 8
            sl = spot_p + 100.0
            tp1 = spot_p - 150.0
            tp2 = spot_p - 300.0
            rationale = (f"Futures CVD spiked aggressively (+{fut_5m:.1f} BTC in 5m, z={fut_z:.1f}) "
                         f"while Spot CVD is lagging (+{spot_5m:.1f} BTC). Overleveraged perp buying prone to long squeeze.")

        # RULE 3: Bullish Absorption at Support (Reversal Long)
        elif spot_5m > 20 and fut_5m < -50 and price_trend == "DOWN":
            setup_type = "BULLISH ABSORPTION LONG"
            signal = "LONG"
            confidence = 7
            sl = spot_p - 150.0
            tp1 = spot_p + 180.0
            tp2 = spot_p + 350.0
            rationale = (f"Retail/perps aggressively selling futures (Fut 5m: {fut_5m:.1f} BTC), "
                         f"but Spot buyers are absorbing limit bids (Spot 5m: +{spot_5m:.1f} BTC). Reversal setup.")

        # RULE 4: Perp Capitulation Flush / Extremes (Mean Reversion Long)
        elif fut_1m < -150 and fut_z < -Z_SCORE_THRESHOLD:
            setup_type = "PERP CAPITULATION FLUSH LONG"
            signal = "LONG"
            confidence = 7.5
            sl = spot_p - 180.0
            tp1 = spot_p + 200.0
            tp2 = spot_p + 400.0
            rationale = (f"Extreme negative 1m delta on Futures ({fut_1m:.1f} BTC, z={fut_z:.1f}). "
                         f"Long liquidation flush exhaustion. High-probability bounce scalp.")

        # RULE 5: Bearish Absorption / Spot Selling (Short Scalp)
        elif spot_5m < -40 and fut_5m > 0 and price_trend != "UP":
            setup_type = "BEARISH SPOT DISTRIBUTION SHORT"
            signal = "SHORT"
            confidence = 7.5
            sl = spot_p + 120.0
            tp1 = spot_p - 160.0
            tp2 = spot_p - 320.0
            rationale = (f"Heavy spot selling distribution (Spot 5m: {spot_5m:.1f} BTC) "
                         f"while perps try to buy. Institutional distribution in progress.")

        if signal:
            risk = abs(spot_p - sl)
            reward = abs(tp1 - spot_p)
            rr = reward / risk if risk > 0 else 1.0
            return {
                "signal": signal,
                "setup_type": setup_type,
                "confidence": confidence,
                "price": spot_p,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr": rr,
                "rationale": rationale,
                "spot_1m": spot_1m,
                "fut_1m": fut_1m,
                "spot_5m": spot_5m,
                "fut_5m": fut_5m,
                "basis": basis,
                "basis_bps": basis_bps
            }
        return None


# ==========================================
# TERMINAL DASHBOARD UI
# ==========================================
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def render_dashboard(monitor: BinanceCVDMonitor, signal_engine: ScalpSignalEngine):
    clear_screen()
    
    spot_p = monitor.spot_price
    fut_p = monitor.futures_price
    basis = monitor.basis
    basis_bps = monitor.basis_bps

    spot_cvd_btc = monitor.spot_cvd_btc
    fut_cvd_btc = monitor.futures_cvd_btc

    spot_1m, _, _ = monitor.get_window_delta(monitor.spot_trades, 60)
    fut_1m, _, _ = monitor.get_window_delta(monitor.futures_trades, 60)
    spot_5m, _, _ = monitor.get_window_delta(monitor.spot_trades, 300)
    fut_5m, _, _ = monitor.get_window_delta(monitor.futures_trades, 300)
    spot_15m, _, _ = monitor.get_window_delta(monitor.spot_trades, 900)
    fut_15m, _, _ = monitor.get_window_delta(monitor.futures_trades, 900)

    print("=" * 85)
    print(f"  BTCUSDT BINANCE SPOT & FUTURES CVD SCALPING MONITOR  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 85)
    print(f"  Spot Price    : ${spot_p:,.2f}          | Futures Price : ${fut_p:,.2f}")
    print(f"  Basis Spread  : ${basis:+,.2f} ({basis_bps:+.2f} bps)  | Status        : ACTIVE (Live Websocket)")
    print("-" * 85)
    print(f"  [CUMULATIVE VOLUME DELTA (CVD)]")
    print(f"    Spot CVD    : {spot_cvd_btc:+10.2f} BTC  (${monitor.spot_cvd_usd:+12,.0f} USD)")
    print(f"    Futures CVD : {fut_cvd_btc:+10.2f} BTC  (${monitor.futures_cvd_usd:+12,.0f} USD)")
    print("-" * 85)
    print(f"  [ORDER FLOW DELTA WINDOWS]")
    print(f"    {'Timeframe':<10} | {'Spot Delta (BTC)':<18} | {'Futures Delta (BTC)':<20} | {'Divergence Note':<25}")
    print(f"    {'-'*10} | {'-'*18} | {'-'*20} | {'-'*25}")
    print(f"    {'1 Minute':<10} | {spot_1m:+18.2f} | {fut_1m:+20.2f} | {'⚡ Fast Momentum' if abs(fut_1m)>50 else 'Normal':<25}")
    print(f"    {'5 Minutes':<10} | {spot_5m:+18.2f} | {fut_5m:+20.2f} | {'🔥 Divergence Alert' if (spot_5m * fut_5m < 0) else 'Aligned':<25}")
    print(f"    {'15 Minutes':<10} | {spot_15m:+18.2f} | {fut_15m:+20.2f} | {'📊 Trend Flow' if abs(spot_15m)>100 else 'Consolidating':<25}")
    print("=" * 85)

    # Evaluate Signal
    setup = signal_engine.evaluate()
    if setup:
        print(f"\n  🚨 [SCALPING OPPORTUNITY SUGGESTION] 🚨")
        print(f"    Setup Type  : {setup['setup_type']}")
        print(f"    Action      : {setup['signal']} @ ${setup['price']:,.2f}")
        print(f"    Confidence  : {setup['confidence']} / 10  (R:R ~ {setup['rr']:.2f})")
        print(f"    Stop Loss   : ${setup['sl']:,.2f}")
        print(f"    Take Profit : TP1: ${setup['tp1']:,.2f}  |  TP2: ${setup['tp2']:,.2f}")
        print(f"    Rationale   : {setup['rationale']}")
        print("-" * 85)

        # Send Telegram alert if cooldown passed (e.g. 3 mins)
        now = time.time()
        if now - monitor.last_signal_time > 180:
            monitor.last_signal_time = now
            msg = (
                f"🚨 *BTC SCALP SETUP: {setup['setup_type']}*\n\n"
                f"*Action:* `{setup['signal']}`\n"
                f"*Entry:* `${setup['price']:,.2f}`\n"
                f"*Stop Loss:* `${setup['sl']:,.2f}`\n"
                f"*Take Profit 1:* `${setup['tp1']:,.2f}`\n"
                f"*R:R Ratio:* `{setup['rr']:.2f}`\n"
                f"*Confidence:* `{setup['confidence']}/10`\n\n"
                f"*Rationale:* {setup['rationale']}"
            )
            send_telegram_alert(msg)
    else:
        print(f"\n  ⏳ Scanning order flow for high-probability scalping setups...")
        print(f"     (Waiting for delta divergence, CVD spikes, or absorption signals)")

    print("=" * 85)
    print("  Press Ctrl+C to exit monitor.")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Initializing Binance Spot & Futures CVD Scalper Monitor...")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(" 📲 TELEGRAM NOTIFICATIONS: ENABLED")
        send_telegram_alert(f"🟢 *CVD Scalper Bot Online*\nMonitoring `{SYMBOL}` on Binance Spot & Futures.")
    else:
        print(" ⚠️ TELEGRAM NOTIFICATIONS: DISABLED (Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in environment/.env)")

    monitor = BinanceCVDMonitor(SYMBOL)
    monitor.start()

    signal_engine = ScalpSignalEngine(monitor)

    # Wait for initial connection data
    time.sleep(2.0)

    try:
        while True:
            render_dashboard(monitor, signal_engine)
            time.sleep(REFRESH_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nStopping monitor gracefully...")
        monitor.stop()
        print("Done.")

if __name__ == "__main__":
    main()

import asyncio
import io
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file into os.environ

import aiohttp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import optional Deribit Macro Synthesizer ─────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from deribit_metrics import DeribitSynthesizer
except ImportError:
    DeribitSynthesizer = None

# ── Timezone ──────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def ist(dt: datetime) -> datetime:
    return dt.astimezone(IST)


def ist_str(dt: datetime, fmt: str = "%d-%b %H:%M IST") -> str:
    return ist(dt).strftime(fmt)


def ist_converter(*args):
    return datetime.now(IST).timetuple()


# ── Logging ───────────────────────────────────────────────────────
logging.Formatter.converter = ist_converter
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s IST - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StraddleStandalone")


# ── Config ────────────────────────────────────────────────────────
class Config:
    SYMBOL                = "BTC"
    LOOKBACK_WEEKS        = 52

    # Delta India MV contracts expire at 12:00 UTC = 17:30 IST daily
    ANCHOR_HOUR_UTC       = 12
    ANCHOR_MINUTE_UTC     = 0
    CYCLE_DURATION_H      = 24

    # How many regime-similar weeks to use for seasonal stats
    DVOL_REGIME_SAMPLES   = 14

    # Term-structure adjustment: 1-DTE IV vs 30-day DVOL
    TERM_STRUCTURE_FACTOR = 1.30

    # Minimum realized move to include a week in VRP calc
    MIN_REALIZED_MOVE_PCT = 0.10

    # API endpoints
    DERIBIT_URL         = "https://www.deribit.com/api/v2/public/"
    BINANCE_URLS        = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]
    BINANCE_URL         = BINANCE_URLS[0]
    DELTA_INDIA_BASE    = "https://api.india.delta.exchange/v2"

    # Output channel toggle: set to "telegram" (default) or "discord"
    OUTPUT_CHANNEL = "telegram"

    # Telegram settings (default output target)
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    # Discord Webhook
    DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

    # Paths
    SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH            = os.path.join(SCRIPT_DIR, "ohlcv_cache.csv")
    CACHE_META_PATH     = os.path.join(SCRIPT_DIR, "ohlcv_cache_meta.json")
    PREMIUM_CACHE_PATH  = os.path.join(SCRIPT_DIR, "premium_cache.json")
    RUN_HISTORY_PATH    = os.path.join(SCRIPT_DIR, "run_history.jsonl")
    MTM_CACHE_PATH      = os.path.join(SCRIPT_DIR, "mtm_cache.json")
    CACHE_VERSION       = "v10"

    # Premium cache: entries older than this are re-verified once per session
    PREMIUM_CACHE_MAX_AGE_DAYS = 28

    # Network
    REQUEST_TIMEOUT_SECS = 12
    RETRY_ATTEMPTS       = 3
    RETRY_BACKOFF_BASE   = 2.0

    # Signal thresholds
    EXHAUSTION_THRESHOLD  = 80.0
    EDGE_THRESHOLD_PCT    = 0.30   # % of spot
    MIN_SEASONAL_STD      = 0.30   # % of spot

    # Blackout: Deribit settles at 08:00 UTC; avoid 08–10 UTC
    BLACKOUT_UTC_HOURS    = {8, 9, 10}

    # Intraday spot momentum
    SPOT_MOMENTUM_CANDLES = 4      # last N hourly candles for slope

    # Visualization
    CHART_FIGSIZE = (16, 13)
    CHART_DPI     = 110


# ── Utility ───────────────────────────────────────────────────────
def canonical_anchor(expiry_utc: datetime) -> datetime:
    """
    Canonical anchor = 12:00 UTC on the calendar day before expiry.
    Delta India MV contracts open at 12:00 UTC (17:30 IST) daily.
    """
    prev_day = expiry_utc.date() - timedelta(days=1)
    return datetime(
        prev_day.year, prev_day.month, prev_day.day,
        Config.ANCHOR_HOUR_UTC, Config.ANCHOR_MINUTE_UTC, 0,
        tzinfo=timezone.utc,
    )


def fmt_day(dt: datetime) -> str:
    return dt.strftime("%d")


# ── Models ────────────────────────────────────────────────────────
@dataclass
class MarketSnapshot:
    straddle_expiry:     datetime          = field(default_factory=lambda: datetime.now(timezone.utc))
    straddle_strike:     float             = 0.0
    straddle_iv:         float             = 0.0
    straddle_prem_pct:   float             = 0.0
    straddle_prem_usd:   float             = 0.0
    straddle_index:      float             = 0.0
    day_of_week:         str               = ""
    now_utc:             datetime          = field(default_factory=lambda: datetime.now(timezone.utc))

    # Delta India live MV contract
    delta_symbol:        Optional[str]     = None
    delta_price_usd:     float             = 0.0
    delta_price_pct:     float             = 0.0
    delta_expiry:        Optional[datetime] = None
    delta_available:     bool              = False

    # 25-delta skew
    skew_iv:             float             = 0.0   # put_iv − call_iv
    skew_vol_imbalance:  float             = 0.0   # (call_vol − put_vol) / total
    skew_composite:      float             = 0.0
    skew_available:      bool              = False

    # Gamma pin
    gamma_pin_strike:    Optional[float]   = None
    gamma_pin_oi_pct:    float             = 0.0

    # Spot momentum
    spot_momentum_slope: float             = 0.0   # $/hr
    spot_momentum_label: str               = "flat"

    # Macro
    dealer_score:        Optional[float]   = None


@dataclass
class RegimeWeek:
    """One historical straddle cycle — fully traceable."""
    expiry_utc:       datetime
    anchor_utc:       datetime          # always expiry − 24 h (12:00 UTC previous day)
    elapsed_end_utc:  datetime          # anchor + elapsed_h (apples-to-apples endpoint)

    start_price:      float             # spot at anchor
    end_price:        float             # spot at elapsed_end
    window_high:      float
    window_low:       float

    pct_move_c2c:     float             # |log(end/start)| × 100
    pct_range_hl:     float             # |log(high/low)| × 100
    path_efficiency:  float

    start_dvol:       float             # DVOL at anchor (real, not estimated)
    mv_symbol:        Optional[str]
    mv_price_usd:     float             # real MV or synthetic ATM straddle
    mv_pct:           float             # mv_price / start_price × 100
    is_real_mv:       bool              # True = MV contract, False = synthetic C+P
    mv_mtm_price_usd: float             # Premium at the 'elapsed_end_utc' time
    mv_mtm_pct:       float             # MTM premium as % of start_price
    mtm_source:       str               # "MV" or "C+P" (derived method for MTM)
    mtm_strike:       int               # ATM strike at the point-in-time
    source:           str               # "MV", "SYNTH", "CACHE_MV", "CACHE_SYNTH"

    vrp:              float             # mv_pct / pct_move_c2c  (NaN if move too small)
    dvol_diff:        float             # |start_dvol − current_dvol|


@dataclass
class RealisedMetrics:
    regime_weeks:        List[RegimeWeek] = field(default_factory=list)
    all_weeks:           List[RegimeWeek] = field(default_factory=list)

    seasonal_avg_pct:    float          = 0.0
    seasonal_range_avg_pct: float       = 0.0
    seasonal_avg_usd:    float          = 0.0
    seasonal_std_pct:    float          = 0.0
    global_avg_pct:      float          = 0.0

    # Current window
    anchor_utc:          datetime       = field(default_factory=lambda: datetime.now(timezone.utc))
    anchor_price:        float          = 0.0
    current_price:       float          = 0.0
    window_high:         float          = 0.0
    window_low:          float          = 0.0
    move_since_anchor:   float          = 0.0   # max excursion from anchor (exhaustion)
    close_to_close_move: float          = 0.0   # log c2c (percentile rank)
    path_efficiency:     float          = 0.0

    # Intraday spot path from anchor to now: [(timestamp_utc, close_price), ...]
    price_path:          List[Tuple[datetime, float]] = field(default_factory=list)

    # Starting implied from real MV price at anchor
    starting_mv_price:   float          = 0.0
    starting_mv_symbol:  Optional[str]  = None
    starting_implied_pct: float         = 0.0
    regime_exhaustion_avg: float        = 0.0
    regime_mtm_avg_pct:  float          = 0.0

    # DVOL
    current_dvol:        float          = 0.0
    dvol_momentum:       str            = "flat"
    dvol_slope_per_hr:   float          = 0.0

    # VRP
    vrp_mean:            float          = 0.0
    vrp_current:         float          = 0.0
    vrp_sample_count:    int            = 0


@dataclass
class SignalState:
    is_exhausted:           bool    = False
    is_premium_rich:        bool    = False
    is_safe_time:           bool    = True
    safe_to_short:          bool    = False

    conviction_score:       float   = 0.0
    conviction_label:       str     = "WAIT"
    consecutive_count:      int     = 1

    exhaustion_ratio:       float   = 0.0
    regime_exhaustion_avg:  float   = 0.0
    remaining_edge_pct:     float   = 0.0
    remaining_edge_usd:     float   = 0.0
    percentile_rank:        float   = 0.0
    theta_per_hour_usd:     float   = 0.0
    hours_to_expiry:        float   = 0.0
    elapsed_hours:          float   = 0.0

    market_price_pct:       float   = 0.0
    market_price_usd:       float   = 0.0
    deribit_fair_pct:       float   = 0.0
    deribit_fair_usd:       float   = 0.0

    seasonal_avg_pct:       float   = 0.0
    seasonal_avg_usd:       float   = 0.0
    seasonal_range_avg_usd: float   = 0.0
    seasonal_std_pct:       float   = 0.0
    pricing_spread_pct:     float   = 0.0
    strategy_suggestion:    str     = "WAIT"


# ── Network Engine ────────────────────────────────────────────────
async def fetch_json(
    session:  aiohttp.ClientSession,
    url:      str,
    params:   dict = None,
    retries:  int  = Config.RETRY_ATTEMPTS,
    backoff:  float = Config.RETRY_BACKOFF_BASE,
    label:    str  = "",
) -> Optional[dict]:
    timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT_SECS)
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status in (400, 404):
                    return None
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            wait = backoff ** attempt
            if attempt < retries - 1:
                logger.debug(f"[{label or url}] attempt {attempt+1} failed ({exc}); retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[{label or url}] all {retries} attempts failed: {exc}")
    return None


# ── Strategy Recommender ──────────────────────────────────────────
def recommend_strategy(snap: MarketSnapshot, rm: RealisedMetrics, sig: SignalState) -> str:
    """
    Suggests the best options strategy based on current market dynamics and regime analysis.
    """
    label = sig.conviction_label
    exh   = sig.exhaustion_ratio
    skew  = snap.skew_iv if snap.skew_available else 0
    vrp   = rm.vrp_mean

    # --- BUY SIDE (Long Volatility / Long Gamma) ---
    if label == "GAMMA BUY":
        return "LONG STRADDLE (Gamma Play: High exhaustion suggests parabolic volatility potential)"

    if label == "VALUE BUY":
        if sig.conviction_score >= 75:
            return "LONG STRADDLE (Deep Value: IV is heavily underpriced relative to regime RV)"
        return "LONG STRANGLE (Cost-efficient long vol entry)"

    # --- SELL SIDE (Short Volatility / Yield Harvesting) ---
    if "SHORT" in label:
        if skew > 4.5:
            return "SHORT IRON FLY (Skew-biased: Puts are expensive, harvest downside premium with a cap)"
        if skew < -4.5:
            return "BULL PUT SPREAD (Skew-biased: Call premium is rich, sell puts with protection)"

        if label == "STRONG SHORT":
            if exh > 120 and vrp > 1.25:
                return "SHORT STRADDLE (Aggressive: Extreme exhaustion + positive VRP mean reversion)"
            return "IRON BUTTERFLY (Capped risk short vol for tail protection)"

        if label == "SOFT SHORT":
            return "IRON CONDOR (Yield harvest: Selling rich premium in a quiet range)"

    # --- NO TRADE / WAIT ---
    if exh > 150:
        return "WAIT (Blow-off risk: Move is parabolic, wait for peak exhaustion to flip short)"

    return "NO TRADE (Edge or Exhaustion levels insufficient for high-probability entry)"


# ── Signal Engine ─────────────────────────────────────────────────
def _count_consecutive_signals(current_label: str) -> int:
    path = Config.RUN_HISTORY_PATH
    if not os.path.exists(path):
        return 1
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        count = 0
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if rec.get("conviction_label") == current_label:
                    count += 1
                else:
                    break
            except Exception:
                break
        return max(1, count + 1)
    except Exception:
        return 1


def compute_signal(
    snap:               MarketSnapshot,
    rm:                 RealisedMetrics,
    now_utc:            datetime,
    target_expiry_utc:  datetime,
    reference_fair_pct: float,
) -> SignalState:

    hours_to_expiry = (target_expiry_utc - now_utc).total_seconds() / 3600
    elapsed_hours   = (now_utc - rm.anchor_utc).total_seconds() / 3600
    iv_pct          = snap.straddle_iv
    index_price     = snap.straddle_index

    is_safe_time = now_utc.hour not in Config.BLACKOUT_UTC_HOURS

    # ── Theta ──
    T_annual     = max(hours_to_expiry, 0.5) / (365 * 24)
    theta_per_hr = 0.0
    if iv_pct > 0 and index_price > 0 and T_annual > 0:
        sigma         = iv_pct / 100
        theta_vanilla = (index_price * sigma) / (2 * math.sqrt(2 * math.pi * T_annual) * 365 * 24)
        time_fraction = max(0.0, min(1.0, hours_to_expiry / Config.CYCLE_DURATION_H))
        acceleration  = 1.0 + (1.0 - time_fraction) ** 2
        theta_per_hr  = theta_vanilla * acceleration

    market_price_pct = snap.delta_price_pct if snap.delta_available else reference_fair_pct
    market_price_usd = snap.delta_price_usd if snap.delta_available else (reference_fair_pct / 100) * index_price

    seasonal_avg_pct = rm.seasonal_avg_pct
    seasonal_std_pct = rm.seasonal_std_pct
    seasonal_avg_usd = rm.seasonal_avg_usd

    # ── Exhaustion ──
    exhaustion_ratio = (
        (rm.move_since_anchor / rm.starting_implied_pct) * 100
        if rm.starting_implied_pct > 0 else 0.0
    )
    is_exhausted = exhaustion_ratio >= Config.EXHAUSTION_THRESHOLD

    # ── Remaining edge ──
    remaining_edge_pct = market_price_pct - seasonal_avg_pct
    remaining_edge_usd = market_price_usd - seasonal_avg_usd
    is_premium_rich    = remaining_edge_pct >= Config.EDGE_THRESHOLD_PCT

    # ── Percentile rank ──
    if rm.regime_weeks:
        moves    = [w.pct_move_c2c for w in rm.regime_weeks]
        pct_rank = float(np.mean([m < rm.close_to_close_move for m in moves]) * 100)
    else:
        pct_rank = 0.0

    spot_trending = snap.spot_momentum_label in ("trending_up", "trending_down")

    # ── Conviction scoring ──
    if remaining_edge_pct >= 0:
        # SELL side
        score  = 0.0
        score += min(35.0, exhaustion_ratio * 0.35)
        edge_z = remaining_edge_pct / seasonal_std_pct if seasonal_std_pct > 0 else 0.0
        score += min(25.0, max(0.0, edge_z * 8.0))
        if rm.dvol_momentum == "compressing":
            score += 15.0
        elif rm.dvol_momentum == "expanding":
            score -= 10.0
        if not math.isnan(rm.vrp_mean) and rm.vrp_mean > 1.20 and rm.vrp_sample_count >= 6:
            score += 10.0
        if rm.path_efficiency > 0.75:
            score -= 10.0
        if snap.skew_available:
            score += 5.0 if snap.skew_composite > 0.05 else (-5.0 if snap.skew_composite < -0.05 else 0.0)
        if snap.gamma_pin_strike and index_price > 0:
            dist_pct = abs(index_price - snap.gamma_pin_strike) / index_price * 100
            if dist_pct < 0.35 and snap.gamma_pin_oi_pct >= 4.0:
                score += 6.0
        if spot_trending:
            score -= 8.0
        score = max(0.0, min(100.0, score))
        label = (
            "STRONG SHORT" if score >= 80 else
            "SOFT SHORT"   if score >= 60 else
            "NEUTRAL"      if score >= 40 else
            "WAIT"
        )
    else:
        # BUY side
        score  = 0.0
        edge_z = abs(remaining_edge_pct) / seasonal_std_pct if seasonal_std_pct > 0 else 0.0
        score += min(35.0, max(0.0, edge_z * 8.0))
        if exhaustion_ratio >= Config.EXHAUSTION_THRESHOLD:
            score += 20.0
        else:
            score += min(20.0, (1.0 - exhaustion_ratio / 100) * 25.0)
        if rm.dvol_momentum == "expanding":
            score += 15.0
        elif rm.dvol_momentum == "compressing":
            score -= 10.0
        if not math.isnan(rm.vrp_mean) and rm.vrp_mean < 0.90 and rm.vrp_sample_count >= 6:
            score += 10.0
        if rm.path_efficiency > 0.75:
            score += 10.0
        if snap.skew_available and abs(snap.skew_composite) > 0.05:
            score += 5.0
        if spot_trending:
            score += 10.0
        score = max(0.0, min(100.0, score))
        label = (
            "GAMMA BUY" if exhaustion_ratio >= Config.EXHAUSTION_THRESHOLD else
            "VALUE BUY" if score >= 60 else
            "WAIT"
        )

    safe_to_short      = remaining_edge_pct >= 0 and score >= 60 and is_safe_time
    reference_fair_usd = (reference_fair_pct / 100) * index_price
    pricing_spread     = market_price_pct - reference_fair_pct if snap.delta_available else 0.0
    consecutive        = _count_consecutive_signals(label)

    logger.info(
        f"Signal | {'SELL' if remaining_edge_pct >= 0 else 'BUY'} | "
        f"{label} {score:.1f}/100 | Exhaustion {exhaustion_ratio:.1f}% | "
        f"Edge {remaining_edge_pct:+.2f}% (${remaining_edge_usd:+,.0f}) | "
        f"DVOL {rm.dvol_momentum} | VRP {rm.vrp_mean:.2f} (n={rm.vrp_sample_count}) | "
        f"PathEff {rm.path_efficiency:.2f} | Skew {snap.skew_composite:+.3f} | "
        f"Spot {snap.spot_momentum_label} | SafeTime {is_safe_time} | Consecutive {consecutive}"
    )

    sig = SignalState(
        is_exhausted          = is_exhausted,
        is_premium_rich       = is_premium_rich,
        is_safe_time          = is_safe_time,
        safe_to_short         = safe_to_short,
        conviction_score      = score,
        conviction_label      = label,
        consecutive_count     = consecutive,
        exhaustion_ratio      = exhaustion_ratio,
        regime_exhaustion_avg = rm.regime_exhaustion_avg,
        remaining_edge_pct    = remaining_edge_pct,
        remaining_edge_usd    = remaining_edge_usd,
        percentile_rank       = pct_rank,
        theta_per_hour_usd    = theta_per_hr,
        hours_to_expiry       = hours_to_expiry,
        elapsed_hours         = elapsed_hours,
        market_price_pct      = market_price_pct,
        market_price_usd      = market_price_usd,
        deribit_fair_pct      = reference_fair_pct,
        deribit_fair_usd      = reference_fair_usd,
        seasonal_avg_pct      = seasonal_avg_pct,
        seasonal_avg_usd      = seasonal_avg_usd,
        seasonal_range_avg_usd= (rm.seasonal_range_avg_pct / 100) * rm.anchor_price,
        seasonal_std_pct      = seasonal_std_pct,
        pricing_spread_pct    = pricing_spread,
    )

    sig.strategy_suggestion = recommend_strategy(snap, rm, sig)
    return sig


# ── Data Engine ───────────────────────────────────────────────────
class DataEngine:
    """Handles all data fetching, caching, and historical premium building."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.sem = asyncio.Semaphore(8)

    async def fetch_dvol_history(self, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
        all_data = []
        cursor   = start_ts_ms
        chunk    = 1000 * 3_600_000  # 1000 hours per page

        while cursor < end_ts_ms:
            page_end = min(cursor + chunk, end_ts_ms)
            res = await fetch_json(
                self.session,
                f"{Config.DERIBIT_URL}get_volatility_index_data",
                params={
                    "currency"       : Config.SYMBOL,
                    "start_timestamp": cursor,
                    "end_timestamp"  : page_end,
                    "resolution"     : "3600",
                },
                label="Deribit/dvol_history",
            )
            if res and "result" in res and res["result"].get("data"):
                rows = res["result"]["data"]
                all_data.extend(rows)
                if len(rows) < 10:
                    break
                cursor = rows[-1][0] + 1
            else:
                cursor = page_end + 1

        if not all_data:
            logger.warning("DVOL history: no data fetched")
            return pd.DataFrame(columns=["timestamp", "dvol_index"])

        dvol_df = pd.DataFrame(all_data, columns=["ts", "o", "h", "l", "c"])
        dvol_df["timestamp"]  = pd.to_datetime(dvol_df["ts"], unit="ms", utc=True)
        dvol_df["dvol_index"] = dvol_df["c"].astype(float)
        dvol_df = dvol_df[["timestamp", "dvol_index"]].sort_values("timestamp").drop_duplicates("timestamp")
        logger.info(f"DVOL history: {len(dvol_df)} hourly rows fetched (paginated)")
        return dvol_df

    async def get_ohlcv_df(self) -> Optional[pd.DataFrame]:
        REQUIRED = {"timestamp", "close", "open", "high", "low", "dvol_index"}

        if os.path.exists(Config.CSV_PATH) and os.path.exists(Config.CACHE_META_PATH):
            try:
                with open(Config.CACHE_META_PATH) as f:
                    meta = json.load(f)
                if meta.get("version") == Config.CACHE_VERSION:
                    cached = pd.read_csv(Config.CSV_PATH, parse_dates=["timestamp"])
                    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
                    if REQUIRED.issubset(set(cached.columns)):
                        age = datetime.now(timezone.utc) - cached["timestamp"].max()
                        if age < timedelta(hours=1):
                            logger.info(f"OHLCV cache hit (age {age})")
                            return cached
                        else:
                            logger.info(f"OHLCV cache stale ({age}), re-fetching")
            except Exception as exc:
                logger.warning(f"OHLCV cache read error: {exc}")

        return await self._fetch_ohlcv_fresh()

    async def _fetch_ohlcv_fresh(self) -> Optional[pd.DataFrame]:
        weeks     = Config.LOOKBACK_WEEKS
        end_ts    = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts  = end_ts - weeks * 7 * 24 * 3_600_000

        all_res       = []
        current_start = start_ts
        timeout       = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT_SECS)
        active_url    = Config.BINANCE_URLS[0]

        while True:
            params = {"symbol": f"{Config.SYMBOL}USDT", "interval": "1h",
                      "startTime": current_start, "limit": 1000}
            res = None

            # Try active_url first, then fallback to others
            candidates = [active_url] + [u for u in Config.BINANCE_URLS if u != active_url]
            for url in candidates:
                try:
                    async with self.session.get(url, params=params, timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, list) and len(data) > 0:
                                res = data
                                active_url = url
                                break
                            elif isinstance(data, list) and len(data) == 0:
                                # Clean end of data
                                res = []
                                active_url = url
                                break
                        else:
                            err_msg = await resp.text()
                            logger.warning(f"Binance endpoint {url} returned HTTP {resp.status}: {err_msg[:100]}")
                except Exception as exc:
                    logger.warning(f"Binance endpoint {url} error: {exc}")

            if not res or not isinstance(res, list):
                break
            all_res.extend(res)
            current_start = res[-1][0] + 1
            if len(res) < 1000:
                break

        if not all_res:
            logger.error("No Binance OHLCV data from any available endpoint")
            return None

        df = pd.DataFrame(all_res, columns=["ts", "o", "h", "l", "c", "v", "ct", "qa", "tr", "tb", "tq", "i"])
        df["close"]     = df["c"].astype(float)
        df["open"]      = df["o"].astype(float)
        df["high"]      = df["h"].astype(float)
        df["low"]       = df["l"].astype(float)
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

        dvol_df = await self.fetch_dvol_history(start_ts, end_ts)
        if not dvol_df.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                dvol_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
            )
        else:
            df["dvol_index"] = float("nan")

        df.to_csv(Config.CSV_PATH, index=False)
        with open(Config.CACHE_META_PATH, "w") as f:
            json.dump({"version": Config.CACHE_VERSION,
                       "written_utc": datetime.now(timezone.utc).isoformat()}, f)

        dvol_ok = df["dvol_index"].notna().sum()
        logger.info(f"OHLCV fetched: {len(df)} rows | DVOL populated: {dvol_ok}/{len(df)}")
        return df

    def _load_premium_cache(self) -> dict:
        if not os.path.exists(Config.PREMIUM_CACHE_PATH):
            return {}
        try:
            with open(Config.PREMIUM_CACHE_PATH) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Premium cache load error: {exc}")
            return {}

    def _save_premium_cache(self, cache: dict):
        try:
            with open(Config.PREMIUM_CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as exc:
            logger.warning(f"Premium cache save error: {exc}")

    def _cache_key(self, expiry_utc: datetime) -> str:
        return expiry_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _is_cache_stale(self, entry: dict) -> bool:
        written = entry.get("written_utc")
        if not written:
            return True
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(written)
            return age.days > Config.PREMIUM_CACHE_MAX_AGE_DAYS
        except Exception:
            return True

    async def fetch_mv_candle(self, symbol: str, anchor_ts: int) -> Optional[float]:
        """Fetch the first available 1-h candle at or within 12 h after anchor_ts."""
        url    = f"{Config.DELTA_INDIA_BASE}/history/candles"
        params = {
            "symbol"    : symbol,
            "resolution": "1h",
            "start"     : anchor_ts - 300,
            "end"       : anchor_ts + 12 * 3600,
        }
        res = await fetch_json(self.session, url, params=params,
                               label=f"DeltaIndia/candle/{symbol}")
        if res and res.get("result"):
            candles = sorted(res["result"], key=lambda x: x.get("time", 0))
            if candles:
                return float(candles[0]["close"])
        return None

    async def build_premium_history(
        self,
        expiry_dates:   List[datetime],
        anchor_prices:  Dict[datetime, float],
        current_symbol: Optional[str] = None,
    ) -> Dict[datetime, dict]:
        cache    = self._load_premium_cache()
        results: Dict[datetime, dict] = {}
        to_fetch: List[datetime]      = []

        for exp_dt in expiry_dates:
            key = self._cache_key(exp_dt)
            if key in cache:
                entry = cache[key]
                if not self._is_cache_stale(entry):
                    results[exp_dt] = entry
                    continue
                if entry.get("source") in ("MV", "CACHE_MV"):
                    results[exp_dt] = entry
                    logger.debug(f"Premium cache: keeping real MV for {key} (stale but trusted)")
                    continue
                logger.debug(f"Premium cache: re-fetching stale synthetic for {key}")
            to_fetch.append(exp_dt)

        from_cache = len(results)
        if not to_fetch:
            logger.info(f"Premium cache: {from_cache} hits (full history loaded)")
            return results

        logger.info(f"Premium cache: {from_cache} hits, {len(to_fetch)} to fetch")
        fetched = await asyncio.gather(*[
            self._fetch_one_premium(exp_dt, anchor_prices, current_symbol, cache)
            for exp_dt in to_fetch
        ])
        new_entries = 0
        for exp_dt, entry in zip(to_fetch, fetched):
            if entry:
                results[exp_dt] = entry
                cache[self._cache_key(exp_dt)] = entry
                new_entries += 1
        if new_entries:
            self._save_premium_cache(cache)
            logger.info(f"Premium cache: saved {new_entries} new entries")

        logger.info(f"Premium history: {len(results)} total premiums recovered")
        return results

    async def _fetch_one_premium(
        self,
        exp_dt:         datetime,
        anchor_prices:  Dict[datetime, float],
        current_symbol: Optional[str],
        cache:          dict,
    ) -> Optional[dict]:
        anchor_utc  = canonical_anchor(exp_dt)
        anchor_ts   = int(anchor_utc.timestamp())
        spot        = anchor_prices.get(exp_dt)
        if not spot:
            return None

        now_utc     = datetime.now(timezone.utc)
        date_str    = exp_dt.strftime("%d%m%y")
        base_strike = int(round(spot / 500) * 500)
        is_target_expiry = abs((exp_dt - now_utc).total_seconds()) < 48 * 3600

        # 1. Live symbol (current expiry only)
        if current_symbol and is_target_expiry:
            async with self.sem:
                px = await self.fetch_mv_candle(current_symbol, anchor_ts)
            if px:
                entry = {"price_usd": px, "symbol": current_symbol,
                         "source": "MV", "written_utc": now_utc.isoformat()}
                logger.info(f"Premium | {ist_str(exp_dt)} | Live MV {current_symbol} = ${px:,.2f}")
                return entry

        # 2. Delta MV predictive naming
        for delta in [0, -500, 500, -1000, 1000]:
            sym = f"MV-BTC-{base_strike + delta}-{date_str}"
            async with self.sem:
                px = await self.fetch_mv_candle(sym, anchor_ts)
            if px:
                entry = {"price_usd": px, "symbol": sym,
                         "source": "MV", "written_utc": now_utc.isoformat()}
                logger.info(f"Premium | {ist_str(exp_dt)} | Discovered MV {sym} = ${px:,.2f}")
                return entry

        # 3. Synthetic ATM C+P fallback
        c_sym = f"C-BTC-{base_strike}-{date_str}"
        p_sym = f"P-BTC-{base_strike}-{date_str}"
        async with self.sem:
            c_px, p_px = await asyncio.gather(
                self.fetch_mv_candle(c_sym, anchor_ts),
                self.fetch_mv_candle(p_sym, anchor_ts),
            )
        if c_px and p_px:
            total_px = c_px + p_px
            entry = {"price_usd": total_px, "symbol": f"SYNTH-{base_strike}",
                     "source": "MV_SYNTH", "written_utc": now_utc.isoformat()}
            logger.info(f"Premium | {ist_str(exp_dt)} | Constructed Synthetic ATM = ${total_px:,.2f}")
            return entry

        logger.debug(f"Premium | {ist_str(exp_dt)} | NOT FOUND — discarding week")
        return None


# ── Options Engine ────────────────────────────────────────────────
class OptionsEngine:
    """Live market data: Delta MV price, skew, gamma pin, DVOL, spot momentum."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def get_delta_mv_price(self) -> Optional[dict]:
        try:
            spot_res = await fetch_json(
                self.session, f"{Config.DELTA_INDIA_BASE}/tickers/BTCUSD",
                label="DeltaIndia/spot",
            )
            spot_price = 0.0
            if spot_res:
                r = spot_res.get("result", {})
                t = r[0] if isinstance(r, list) else r
                if isinstance(t, dict):
                    spot_price = float(t.get("mark_price", 0))

            prod_res = await fetch_json(
                self.session, f"{Config.DELTA_INDIA_BASE}/products",
                label="DeltaIndia/products",
            )
            if not prod_res or "result" not in prod_res:
                logger.warning("Delta India products: no result")
                return None

            now_ts = datetime.now(timezone.utc).timestamp()
            win_ts = now_ts + 48 * 3600
            mv_prods = []

            for p in prod_res["result"]:
                sym = p.get("symbol", "")
                try:
                    ts_str    = (p.get("settlement_time") or p.get("p_settlement_time") or "").replace("Z", "+00:00")
                    settle_ts = datetime.fromisoformat(ts_str).timestamp()
                    p["_settle_ts"] = settle_ts
                except Exception:
                    continue
                if "BTC" in sym.upper() and now_ts < settle_ts <= win_ts:
                    if any(t in sym.upper() for t in ["MV", "MOVE", "STRD", "SDR"]):
                        mv_prods.append(p)

            if not mv_prods:
                logger.warning("No Delta India BTC MV products in 48 h window")
                return None

            target_ts   = now_ts + 24 * 3600
            best_diff   = min(abs(p["_settle_ts"] - target_ts) for p in mv_prods)
            same_expiry = [p for p in mv_prods if abs(p["_settle_ts"] - target_ts) - best_diff < 1.0]

            def _strike(prod):
                if prod.get("strike_price"):
                    try:
                        return float(prod["strike_price"])
                    except Exception:
                        pass
                nums = re.findall(r"\d+", prod.get("symbol", ""))
                return float(nums[0]) if nums else 0.0

            best = min(same_expiry, key=lambda x: abs(_strike(x) - spot_price)) if spot_price else same_expiry[0]
            expiry_dt = datetime.fromisoformat(
                (best.get("settlement_time") or best.get("p_settlement_time")).replace("Z", "+00:00")
            )
            sym = best["symbol"]

            ticker = await fetch_json(
                self.session, f"{Config.DELTA_INDIA_BASE}/tickers/{sym}",
                label=f"DeltaIndia/ticker/{sym}",
            )
            if not ticker or "result" not in ticker:
                return None
            t = ticker["result"]
            if isinstance(t, list):
                t = t[0]
            if not isinstance(t, dict):
                return None

            mark      = float(t.get("mark_price", 0))
            native_iv = float(t.get("mark_vol", 0)) * 100
            idx       = 0.0
            q = t.get("quotes")
            if isinstance(q, list) and q:
                idx = float(q[0].get("index_price", 0))
            elif isinstance(q, dict):
                idx = float(q.get("index_price", 0))
            if idx == 0.0:
                idx = spot_price
            if idx == 0.0:
                logger.warning("Delta India index_price = 0")
                return None

            pct = (mark / idx) * 100
            logger.info(
                f"FETCH | Delta MV {sym} | mark=${mark:,.2f} | idx=${idx:,.2f} | "
                f"{pct:.3f}% | IV={native_iv:.1f}% | expiry {ist_str(expiry_dt)}"
            )
            return {"symbol": sym, "price_usd": mark, "price_pct": pct, "expiry": expiry_dt, "iv": native_iv}

        except Exception as exc:
            logger.error(f"Delta India error: {exc}", exc_info=True)
            return None

    async def get_delta_spot_price(self) -> float:
        res = await fetch_json(
            self.session, f"{Config.DELTA_INDIA_BASE}/tickers/BTCUSD",
            label="DeltaIndia/spot_price",
        )
        if res:
            r = res.get("result", {})
            t = r[0] if isinstance(r, list) else r
            if isinstance(t, dict):
                return float(t.get("mark_price", 0))
        return 0.0

    async def get_delta_synthetic_straddle(self, spot: float, expiry_utc: datetime) -> Optional[dict]:
        """Fetch live ATM synthetic straddle from Delta if MV is missing."""
        date_str = expiry_utc.strftime("%d%m%y")
        strike   = int(round(spot / 500) * 500)
        c_sym    = f"C-BTC-{strike}-{date_str}"
        p_sym    = f"P-BTC-{strike}-{date_str}"

        c_res, p_res = await asyncio.gather(
            fetch_json(self.session, f"{Config.DELTA_INDIA_BASE}/tickers/{c_sym}"),
            fetch_json(self.session, f"{Config.DELTA_INDIA_BASE}/tickers/{p_sym}"),
        )

        def get_mark(res):
            if not res or "result" not in res:
                return None
            t     = res["result"][0] if isinstance(res["result"], list) else res["result"]
            price = float(t.get("mark_price", 0))
            iv    = float(t.get("mark_vol", 0)) * 100
            return price, iv

        c_data, p_data = get_mark(c_res), get_mark(p_res)
        if c_data and p_data:
            c_px, c_iv = c_data
            p_px, p_iv = p_data
            total      = c_px + p_px
            mean_iv    = (c_iv + p_iv) / 2
            return {
                "symbol": f"SYNTH-{strike}", "price_usd": total,
                "price_pct": (total / spot) * 100, "expiry": expiry_utc, "iv": mean_iv,
            }
        return None

    async def get_25d_skew(self, expiry_date: datetime, index_price: float, iv_annual_pct: float) -> dict:
        _empty = {"skew_iv": 0.0, "skew_vol_imbalance": 0.0, "skew_composite": 0.0,
                  "put_iv": 0.0, "call_iv": 0.0, "available": False}
        try:
            T_years = max(
                (expiry_date - datetime.now(timezone.utc)).total_seconds() / (365 * 24 * 3600),
                2 / (365 * 24),
            )
            sigma      = iv_annual_pct / 100.0
            # 0.6745 = z-score for the 25-delta point of a standard normal distribution
            otm_factor = max(0.6745 * sigma * math.sqrt(T_years), 0.005)

            c_strike = round(index_price * math.exp(+otm_factor) / 500) * 500
            p_strike = round(index_price * math.exp(-otm_factor) / 500) * 500
            day_str  = f"{fmt_day(expiry_date)}{expiry_date.strftime('%b%y').upper()}"

            async def _ticker(strike, side):
                name = f"{Config.SYMBOL}-{day_str}-{int(strike)}-{side}"
                r    = await fetch_json(
                    self.session, f"{Config.DERIBIT_URL}ticker",
                    params={"instrument_name": name}, label=f"Deribit/skew/{name}",
                )
                if r and r.get("result"):
                    res = r["result"]
                    iv  = float(res.get("mark_iv", 0))
                    vol = float(res.get("stats", {}).get("volume", 0))
                    return (iv if iv > 0 else None), vol
                return None, 0.0

            (p_iv, p_vol), (c_iv, c_vol) = await asyncio.gather(
                _ticker(p_strike, "P"), _ticker(c_strike, "C")
            )
            if p_iv is None or c_iv is None or c_iv == 0:
                return _empty

            iv_skew       = p_iv - c_iv
            total_vol     = (p_vol or 0.0) + (c_vol or 0.0)
            vol_imbalance = ((c_vol or 0.0) - (p_vol or 0.0)) / total_vol if total_vol > 0 else 0.0
            composite     = 0.70 * (iv_skew / 10.0) + 0.30 * vol_imbalance

            logger.info(
                f"25d Skew | put_iv={p_iv:.1f}% call_iv={c_iv:.1f}% | "
                f"skew={iv_skew:+.2f}% | vol_imb={vol_imbalance:+.2f} | comp={composite:+.3f}"
            )
            return {
                "skew_iv": iv_skew, "skew_vol_imbalance": vol_imbalance,
                "skew_composite": composite, "put_iv": p_iv, "call_iv": c_iv, "available": True,
            }
        except Exception as exc:
            logger.warning(f"25d skew error: {exc}")
            return _empty

    async def get_gamma_pin(self, expiry_date: datetime, index_price: float) -> dict:
        _empty = {"pin_strike": None, "oi_pct": 0.0, "available": False}
        try:
            res = await fetch_json(
                self.session, f"{Config.DERIBIT_URL}get_book_summary_by_currency",
                params={"currency": Config.SYMBOL, "kind": "option"},
                label="Deribit/book_summary",
            )
            if not res or "result" not in res:
                return _empty
            expiry_ms = int(expiry_date.timestamp() * 1000)
            all_expiries = {s.get("expiration_timestamp") for s in res["result"] if s.get("expiration_timestamp")}
            valid_expiries = [ts for ts in all_expiries if abs(ts - expiry_ms) <= 4 * 3600 * 1000]

            if not valid_expiries:
                logger.warning(f"No Deribit expiries found within 4h of {ist_str(expiry_date)}")
                return _empty

            target_ms = min(valid_expiries, key=lambda x: abs(x - expiry_ms))
            summaries = [
                s for s in res["result"]
                if s.get("expiration_timestamp") == target_ms and s.get("open_interest", 0) > 0
            ]
            if not summaries:
                return _empty
            total_oi     = sum(s["open_interest"] for s in summaries)
            oi_by_strike = {}
            for s in summaries:
                st = s.get("strike", 0)
                oi_by_strike[st] = oi_by_strike.get(st, 0) + s["open_interest"]
            pin_strike = max(oi_by_strike, key=oi_by_strike.get)
            oi_pct     = oi_by_strike[pin_strike] / total_oi * 100
            return {"pin_strike": pin_strike, "oi_pct": oi_pct, "available": True}
        except Exception as exc:
            logger.warning(f"Gamma pin error: {exc}")
            return _empty

    async def get_live_dvol(self) -> float:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        res = await fetch_json(
            self.session, f"{Config.DERIBIT_URL}get_volatility_index_data",
            params={
                "currency"       : Config.SYMBOL,
                "start_timestamp": now_ms - 3_600_000,
                "end_timestamp"  : now_ms,
                "resolution"     : "3600",
            },
            label="Deribit/dvol_live",
        )
        if res and "result" in res and res["result"].get("data"):
            return float(res["result"]["data"][-1][4])
        return 0.0

    async def get_spot_momentum(self) -> dict:
        n      = Config.SPOT_MOMENTUM_CANDLES + 1
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        params = {"symbol": f"{Config.SYMBOL}USDT", "interval": "1h",
                  "startTime": now_ms - n * 3_600_000, "limit": n}
        timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT_SECS)
        res = None
        for url in Config.BINANCE_URLS:
            try:
                async with self.session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and len(data) >= 3:
                            res = data
                            break
            except Exception as exc:
                logger.debug(f"Spot momentum Binance endpoint {url} failed: {exc}")

        if not res or len(res) < 3:
            return {"slope": 0.0, "label": "flat"}

        closes = np.array([float(c[4]) for c in res[-Config.SPOT_MOMENTUM_CANDLES:]])
        slope  = float(np.polyfit(np.arange(len(closes)), closes, 1)[0])
        label  = "trending_up" if slope > 50 else ("trending_down" if slope < -50 else "flat")
        logger.info(f"Spot momentum | slope={slope:+.1f} $/hr | {label}")
        return {"slope": slope, "label": label}


# ── Realised Metrics Engine ───────────────────────────────────────
class RealisedMetricsEngine:

    def __init__(self, session: aiohttp.ClientSession):
        self.data = DataEngine(session)
        self.mtm_cache_path = Config.MTM_CACHE_PATH
        self.mtm_cache = self._load_mtm_cache()

    def _load_mtm_cache(self) -> dict:
        if os.path.exists(self.mtm_cache_path):
            try:
                with open(self.mtm_cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_mtm_cache(self):
        try:
            with open(self.mtm_cache_path, "w") as f:
                json.dump(self.mtm_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save MTM cache: {e}")

    async def compute(
        self,
        now_utc:           datetime,
        target_expiry_utc: datetime,
        hours_to_expiry:   float,
        current_dvol:      float,
        current_symbol:    Optional[str] = None,
    ) -> RealisedMetrics:

        df = await self.data.get_ohlcv_df()
        if df is None or df.empty:
            logger.error("No OHLCV data — empty metrics")
            return RealisedMetrics()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── DVOL momentum (last 24 h) ──
        valid_dvol        = df.dropna(subset=["dvol_index"])
        dvol_series       = valid_dvol["dvol_index"].values[-24:]
        dvol_momentum     = "flat"
        dvol_slope_per_hr = 0.0
        if len(dvol_series) >= 3:
            slope = np.polyfit(np.arange(len(dvol_series)), dvol_series, 1)[0]
            dvol_slope_per_hr = float(slope)
            dvol_momentum = "compressing" if slope < -0.05 else ("expanding" if slope > 0.05 else "flat")

        # ── Canonical anchor for today ──
        anchor_utc    = canonical_anchor(target_expiry_utc)
        elapsed_hours = max(0.0, min(
            (now_utc - anchor_utc).total_seconds() / 3600,
            Config.CYCLE_DURATION_H,
        ))

        # ── Historical expiry dates (strict same day-of-week) ──
        today_dow     = target_expiry_utc.weekday()
        hist_expiries: List[datetime] = []
        for week_offset in range(1, Config.LOOKBACK_WEEKS + 1):
            past_exp = target_expiry_utc - timedelta(weeks=week_offset)
            if past_exp.weekday() != today_dow:
                continue
            past_exp = past_exp.replace(hour=12, minute=0, second=0, microsecond=0)
            hist_expiries.append(past_exp)

        # ── Anchor prices for all historical windows ──
        anchor_prices: Dict[datetime, float] = {}
        for exp_dt in hist_expiries:
            anc   = canonical_anchor(exp_dt)
            cands = df[df["timestamp"] <= anc]
            if not cands.empty:
                anchor_prices[exp_dt] = float(cands.iloc[-1]["close"])

        anc_cands = df[df["timestamp"] <= anchor_utc]
        anchor_prices[target_expiry_utc] = float(anc_cands.iloc[-1]["close"]) if not anc_cands.empty else 0.0

        # ── Fetch all premiums ──
        premium_history = await self.data.build_premium_history(
            hist_expiries + [target_expiry_utc], anchor_prices, current_symbol
        )

        # ── Build RegimeWeek objects ──
        all_weeks: List[RegimeWeek] = []
        for exp_dt in hist_expiries:
            anc         = canonical_anchor(exp_dt)
            elapsed_end = anc + timedelta(hours=elapsed_hours)

            cands_start = df[df["timestamp"] <= anc]
            cands_end   = df[df["timestamp"] <= elapsed_end]
            if cands_start.empty or cands_end.empty:
                continue

            start_price = float(cands_start.iloc[-1]["close"])
            end_price   = float(cands_end.iloc[-1]["close"])
            mtm_strike  = int(round(end_price / 500) * 500)
            if start_price <= 0 or end_price <= 0:
                continue

            start_dvol_raw = cands_start.iloc[-1].get("dvol_index")
            if start_dvol_raw is None or (isinstance(start_dvol_raw, float) and math.isnan(start_dvol_raw)):
                start_dvol = float("nan")
            else:
                start_dvol = float(start_dvol_raw)

            window_df = df[(df["timestamp"] >= anc) & (df["timestamp"] <= elapsed_end)]
            wh = window_df["high"].max() if not window_df.empty else start_price
            wl = window_df["low"].min()  if not window_df.empty else start_price

            pct_c2c  = abs(math.log(end_price / start_price)) * 100
            pct_hl   = abs(math.log(wh / wl)) * 100 if wl > 0 else pct_c2c
            hl_usd   = wh - wl
            oc_usd   = abs(end_price - start_price)
            path_eff = float(oc_usd / hl_usd) if hl_usd > 0 else 0.0

            prem = premium_history.get(exp_dt)
            if not prem:
                continue

            mv_price_usd = float(prem["price_usd"])
            mv_symbol    = prem["symbol"]
            source       = prem["source"]
            is_real_mv   = source in ("MV", "CACHE_MV")
            mv_pct       = (mv_price_usd / start_price) * 100 if start_price > 0 else 0.0

            mv_mtm_price = 0.0
            ts_end       = int(elapsed_end.timestamp())
            mtm_source   = "N/A"

            # 1. Real MV symbol fetch
            if is_real_mv:
                async with self.data.sem:
                    px = await self.data.fetch_mv_candle(mv_symbol, ts_end)
                if px:
                    mv_mtm_price = float(px)
                    mtm_source   = "MV"

            # 2. Synthetic Fallback
            if mv_mtm_price <= 0:
                try:
                    nums = [s for s in re.findall(r'\d+', mv_symbol) if len(s) >= 4]
                    if nums:
                        strike   = int(nums[0])
                        date_str = exp_dt.strftime("%d%m%y")
                        c_sym    = f"C-BTC-{strike}-{date_str}"
                        p_sym    = f"P-BTC-{strike}-{date_str}"
                        async with self.data.sem:
                            c_px, p_px = await asyncio.gather(
                                self.data.fetch_mv_candle(c_sym, ts_end),
                                self.data.fetch_mv_candle(p_sym, ts_end),
                            )
                        if c_px and p_px:
                            mv_mtm_price = float(c_px) + float(p_px)
                            mtm_source   = "C+P"
                except Exception:
                    pass

            mv_mtm_pct = (mv_mtm_price / start_price) * 100 if start_price > 0 else 0.0

            vrp = (
                min(mv_pct / pct_c2c, 5.0)
                if pct_c2c >= Config.MIN_REALIZED_MOVE_PCT else float("nan")
            )
            dvol_diff = abs(start_dvol - current_dvol) if not math.isnan(start_dvol) else 999.0

            all_weeks.append(RegimeWeek(
                expiry_utc       = exp_dt,
                anchor_utc       = anc,
                elapsed_end_utc  = elapsed_end,
                start_price      = start_price,
                end_price        = end_price,
                window_high      = wh,
                window_low       = wl,
                pct_move_c2c     = pct_c2c,
                pct_range_hl     = pct_hl,
                path_efficiency  = path_eff,
                start_dvol       = start_dvol,
                mv_symbol        = mv_symbol,
                mv_price_usd     = mv_price_usd,
                mv_pct           = mv_pct,
                is_real_mv       = is_real_mv,
                mv_mtm_price_usd = mv_mtm_price,
                mv_mtm_pct       = mv_mtm_pct,
                mtm_source       = mtm_source,
                mtm_strike       = mtm_strike,
                source           = source,
                vrp              = vrp,
                dvol_diff        = dvol_diff,
            ))

        logger.info(f"Historical windows built: {len(all_weeks)} valid out of {len(hist_expiries)}")

        # ── DVOL regime filter ──
        weeks_with_dvol = sorted(
            [w for w in all_weeks if not math.isnan(w.start_dvol)],
            key=lambda w: w.dvol_diff,
        )
        weeks_no_dvol = [w for w in all_weeks if math.isnan(w.start_dvol)]
        regime_weeks  = weeks_with_dvol[:Config.DVOL_REGIME_SAMPLES]

        if len(regime_weeks) < 6 and weeks_no_dvol:
            regime_weeks += weeks_no_dvol[:max(0, 6 - len(regime_weeks))]
            logger.warning(f"Only {len(regime_weeks)} regime weeks — padded with no-DVOL weeks")

        # ── Regime Exhaustion (historical average) ──
        regime_exh_vals = [
            (w.pct_range_hl / w.mv_pct * 100)
            for w in regime_weeks if w.mv_pct > 0
        ]
        regime_exhaustion_avg = float(np.mean(regime_exh_vals)) if regime_exh_vals else 0.0

        regime_mtm_vals = [
            w.mv_mtm_pct for w in regime_weeks if w.mv_mtm_pct > 0
        ]
        regime_mtm_avg_pct = float(np.mean(regime_mtm_vals)) if regime_mtm_vals else 0.0

        valid_dvols_rw = [w.start_dvol for w in regime_weeks if not math.isnan(w.start_dvol)]
        logger.info(
            f"Regime filter: {len(regime_weeks)} weeks | "
            f"DVOL range: {min(valid_dvols_rw, default=0):.1f}%–{max(valid_dvols_rw, default=0):.1f}%"
        )

        # ── Seasonal stats ──
        global_avg_pct = float(np.mean([w.pct_move_c2c for w in all_weeks])) if all_weeks else 0.0
        if regime_weeks:
            moves                  = np.array([w.pct_move_c2c for w in regime_weeks])
            rngs                   = np.array([w.pct_range_hl  for w in regime_weeks])
            seasonal_avg_pct       = float(np.mean(moves))
            seasonal_range_avg_pct = float(np.mean(rngs))
            raw_std                = float(np.std(moves, ddof=1)) if len(moves) > 1 else 0.0
            seasonal_std_pct       = max(raw_std, Config.MIN_SEASONAL_STD)
        else:
            seasonal_avg_pct       = global_avg_pct
            seasonal_range_avg_pct = global_avg_pct
            seasonal_std_pct       = Config.MIN_SEASONAL_STD

        # ── VRP ──
        vrp_vals      = [w.vrp for w in regime_weeks if not math.isnan(w.vrp)]
        vrp_mean      = float(np.mean(vrp_vals)) if vrp_vals else float("nan")
        vrp_sample_ct = len(vrp_vals)

        # ── Current window ──
        anchor_cands  = df[df["timestamp"] <= anchor_utc]
        now_cands     = df[df["timestamp"] <= now_utc]
        anchor_price  = float(anchor_cands.iloc[-1]["close"]) if not anchor_cands.empty else 0.0
        current_price = float(now_cands.iloc[-1]["close"]) if not now_cands.empty else 0.0

        wh = wl = anchor_price
        move_since_anchor = close_to_close_move = path_efficiency = 0.0
        price_path: List[Tuple[datetime, float]] = []

        if anchor_price > 0 and current_price > 0:
            close_to_close_move = abs(math.log(current_price / anchor_price)) * 100
            window_df = df[(df["timestamp"] >= anchor_utc) & (df["timestamp"] <= now_utc)]
            if not window_df.empty:
                wh  = window_df["high"].max()
                wl  = window_df["low"].min()
                hl  = wh - wl
                farthest = wh if abs(wh - anchor_price) > abs(wl - anchor_price) else wl
                move_since_anchor = abs(math.log(farthest / anchor_price)) * 100 if anchor_price > 0 else 0.0
                path_efficiency   = float(abs(current_price - anchor_price) / hl) if hl > 0 else 0.0
                price_path = list(zip(
                    window_df["timestamp"].tolist(),
                    window_df["close"].astype(float).tolist(),
                ))

        # ── Starting implied ──
        curr_prem = premium_history.get(target_expiry_utc)
        if curr_prem and anchor_price > 0:
            starting_mv_price    = float(curr_prem["price_usd"])
            starting_mv_symbol   = curr_prem["symbol"]
            starting_implied_pct = (starting_mv_price / anchor_price) * 100
            logger.info(
                f"Starting implied | {starting_mv_symbol} "
                f"${starting_mv_price:,.2f} / ${anchor_price:,.0f} = {starting_implied_pct:.2f}%"
            )
        else:
            starting_mv_price    = 0.0
            starting_mv_symbol   = None
            starting_implied_pct = 0.0
            logger.warning("No current MV/synthetic premium at anchor — exhaustion will be 0")

        vrp_current = (
            (starting_implied_pct / close_to_close_move)
            if close_to_close_move > 0 and starting_implied_pct > 0 else float("nan")
        )

        logger.info(
            f"Anchor {ist_str(anchor_utc)} → Now {ist_str(now_utc)} | "
            f"Elapsed {elapsed_hours:.1f}h | C2C {close_to_close_move:.2f}% | "
            f"MaxRange {move_since_anchor:.2f}% | StartingImplied {starting_implied_pct:.2f}% | "
            f"SeasonalAvg {seasonal_avg_pct:.2f}% | DVOL {current_dvol:.1f}% ({dvol_momentum})"
        )

        self._save_mtm_cache()

        return RealisedMetrics(
            regime_weeks           = regime_weeks,
            all_weeks              = all_weeks,
            seasonal_avg_pct       = seasonal_avg_pct,
            seasonal_range_avg_pct = seasonal_range_avg_pct,
            seasonal_avg_usd       = (seasonal_avg_pct / 100) * anchor_price,
            seasonal_std_pct       = seasonal_std_pct,
            global_avg_pct         = global_avg_pct,
            anchor_utc             = anchor_utc,
            anchor_price           = anchor_price,
            current_price          = current_price,
            window_high            = wh,
            window_low             = wl,
            move_since_anchor      = move_since_anchor,
            close_to_close_move    = close_to_close_move,
            path_efficiency        = path_efficiency,
            price_path             = price_path,
            starting_mv_price      = starting_mv_price,
            starting_mv_symbol     = starting_mv_symbol,
            starting_implied_pct   = starting_implied_pct,
            regime_exhaustion_avg  = regime_exhaustion_avg,
            regime_mtm_avg_pct     = regime_mtm_avg_pct,
            current_dvol           = current_dvol,
            dvol_momentum          = dvol_momentum,
            dvol_slope_per_hr      = dvol_slope_per_hr,
            vrp_mean               = vrp_mean,
            vrp_current            = vrp_current,
            vrp_sample_count       = vrp_sample_ct,
        )


# ── Run History Persistence ───────────────────────────────────────
def append_run_history(
    snap:              MarketSnapshot,
    rm:                RealisedMetrics,
    sig:               SignalState,
    target_expiry_utc: datetime,
):
    record = {
        "ts_utc"               : snap.now_utc.isoformat(),
        "anchor_utc"           : rm.anchor_utc.isoformat(),
        "target_expiry_utc"    : target_expiry_utc.isoformat(),
        "elapsed_hours"        : sig.elapsed_hours,
        "hours_to_expiry"      : sig.hours_to_expiry,
        "day_of_week"          : snap.day_of_week,
        "spot"                 : snap.straddle_index,
        "anchor_price"         : rm.anchor_price,
        "strike"               : snap.straddle_strike,
        "iv_pct"               : snap.straddle_iv,
        "current_dvol"         : rm.current_dvol,
        "dvol_momentum"        : rm.dvol_momentum,
        "dvol_slope"           : rm.dvol_slope_per_hr,
        "move_maxrange_pct"    : rm.move_since_anchor,
        "move_c2c_pct"         : rm.close_to_close_move,
        "starting_implied_pct" : rm.starting_implied_pct,
        "starting_mv_price"    : rm.starting_mv_price,
        "starting_mv_symbol"   : rm.starting_mv_symbol,
        "exhaustion_ratio"     : sig.exhaustion_ratio,
        "regime_exhaustion_avg": sig.regime_exhaustion_avg,
        "percentile_rank"      : sig.percentile_rank,
        "path_efficiency"      : rm.path_efficiency,
        "spot_momentum_label"  : snap.spot_momentum_label,
        "spot_momentum_slope"  : snap.spot_momentum_slope,
        "seasonal_avg_pct"     : sig.seasonal_avg_pct,
        "seasonal_std_pct"     : sig.seasonal_std_pct,
        "seasonal_avg_usd"     : sig.seasonal_avg_usd,
        "remaining_edge_pct"   : sig.remaining_edge_pct,
        "remaining_edge_usd"   : sig.remaining_edge_usd,
        "strategy_suggestion"  : sig.strategy_suggestion,
        "market_price_pct"     : sig.market_price_pct,
        "market_price_usd"     : sig.market_price_usd,
        "vrp_mean"             : rm.vrp_mean,
        "vrp_sample_count"     : rm.vrp_sample_count,
        "vrp_current"          : rm.vrp_current,
        "conviction_score"     : sig.conviction_score,
        "conviction_label"     : sig.conviction_label,
        "consecutive_count"    : sig.consecutive_count,
        "delta_available"      : snap.delta_available,
        "delta_symbol"         : snap.delta_symbol,
        "skew_iv"              : snap.skew_iv if snap.skew_available else None,
        "skew_composite"       : snap.skew_composite if snap.skew_available else None,
        "gamma_pin_strike"     : snap.gamma_pin_strike,
        "gamma_pin_oi_pct"     : snap.gamma_pin_oi_pct,
        "theta_per_hr_usd"     : sig.theta_per_hour_usd,
        "is_exhausted"         : sig.is_exhausted,
        "is_premium_rich"      : sig.is_premium_rich,
        "is_safe_time"         : sig.is_safe_time,
        "regime_weeks_count"   : len(rm.regime_weeks),
        "regime_real_mv_count" : sum(1 for w in rm.regime_weeks if w.is_real_mv),
    }
    try:
        with open(Config.RUN_HISTORY_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Run history appended → {Config.RUN_HISTORY_PATH}")
    except Exception as exc:
        logger.warning(f"Run history write error: {exc}")


# ── Visual Style (single source of truth for colors & thresholds) ─
class VizStyle:
    BG_DARK  = "#0a0e1a"
    BG_PANEL = "#0d1117"
    BG_CARD  = "#111827"
    ACCENT   = "#E91E63"
    GREEN    = "#22c55e"
    RED      = "#ef4444"
    GOLD     = "#f59e0b"
    CYAN     = "#22d3ee"
    PURPLE   = "#a855f7"
    WHITE    = "#f8fafc"
    GREY     = "#6b7280"
    ORANGE   = "#f97316"

    SOFT_SCORE = 60.0
    WEAK_SCORE = 40.0

    @classmethod
    def signal_color(cls, sig: SignalState) -> str:
        if sig.conviction_label == "GAMMA BUY":
            return cls.PURPLE
        if sig.conviction_label == "VALUE BUY":
            return cls.CYAN
        if sig.conviction_score >= cls.SOFT_SCORE:
            return cls.GREEN
        if sig.conviction_score >= cls.WEAK_SCORE:
            return cls.GOLD
        return cls.RED

    @classmethod
    def dvol_color(cls, momentum: str) -> str:
        return {"expanding": cls.RED, "compressing": cls.GREEN}.get(momentum, cls.GOLD)

    @classmethod
    def vrp_color(cls, vrp_mean: float) -> str:
        if math.isnan(vrp_mean):
            return cls.GREY
        if vrp_mean > 1.15:
            return cls.GREEN
        if vrp_mean < 0.90:
            return cls.RED
        return cls.GOLD

    @classmethod
    def momentum_color(cls, label: str) -> str:
        return {"trending_up": cls.ORANGE, "trending_down": cls.CYAN}.get(label, cls.WHITE)

    @classmethod
    def edge_color(cls, edge_pct: float) -> str:
        return cls.GREEN if edge_pct > 0 else cls.RED

    @classmethod
    def exhaustion_color(cls, exh_ratio: float) -> str:
        if exh_ratio > Config.EXHAUSTION_THRESHOLD:
            return cls.RED
        if exh_ratio > 60:
            return cls.GOLD
        return cls.GREEN


# ── Straddle Visualizer ───────────────────────────────────────────
class StraddleVisualizer:
    """
    Renders a 4-panel decision dashboard. Each panel answers a distinct
    question — no two panels re-derive the same rich/cheap × exhausted/quiet
    judgment, and no panel uses a dual (twin) y-axis.

    1. Spot Price vs Priced-In Move  — what actually happened to price today,
       against the straddle's implied breakeven band.
    2. Decision: Edge vs Exhaustion  — the single actionable quadrant chart
       that maps directly onto the conviction label / strategy suggestion.
    3. Realized vs Implied ($) by Week — grouped bars, historical comparison.
    4. Volatility Risk Premium Trend — VRP on its own axis, over time.
    """

    S = VizStyle

    @staticmethod
    def generate_report(snap: MarketSnapshot, rm: RealisedMetrics, sig: SignalState) -> io.BytesIO:
        C = StraddleVisualizer.S
        plt.style.use("dark_background")
        fig = plt.figure(figsize=Config.CHART_FIGSIZE)
        fig.set_facecolor(C.BG_DARK)

        sig_col    = C.signal_color(sig)
        consec_sfx = f" ×{sig.consecutive_count}" if sig.consecutive_count > 1 else ""

        fig.text(
            0.5, 0.975,
            f"BTC 1-DTE Straddle  |  {snap.day_of_week}  |  Strike ${snap.straddle_strike:,.0f}",
            ha="center", va="top", fontsize=15, weight="bold", color=C.WHITE,
        )
        fig.text(
            0.5, 0.953,
            f"▶  {sig.conviction_label}{consec_sfx}  ({sig.conviction_score:.0f}/100)   ·   {sig.strategy_suggestion}",
            ha="center", va="top", fontsize=11, weight="bold", color=sig_col,
        )

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[0.55, 3.2, 3.2],
            left=0.07, right=0.96, top=0.92, bottom=0.05,
            hspace=0.50, wspace=0.28,
        )

        ax_kpi = fig.add_subplot(gs[0, :])
        StraddleVisualizer._draw_kpi_banner(ax_kpi, snap, rm, sig)

        ax_price = fig.add_subplot(gs[1, 0])
        StraddleVisualizer._draw_spot_price_panel(ax_price, snap, rm)

        ax_decision = fig.add_subplot(gs[1, 1])
        StraddleVisualizer._draw_decision_quadrant(ax_decision, snap, rm, sig)

        ax_bars = fig.add_subplot(gs[2, 0])
        StraddleVisualizer._draw_regime_bars(ax_bars, rm)

        ax_vrp = fig.add_subplot(gs[2, 1])
        StraddleVisualizer._draw_vrp_trend(ax_vrp, rm)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=Config.CHART_DPI, bbox_inches="tight", facecolor=C.BG_DARK)
        plt.close(fig)
        buf.seek(0)
        return buf

    # ── Panel 0: KPI banner ────────────────────────────────────────
    @staticmethod
    def _draw_kpi_banner(ax, snap: MarketSnapshot, rm: RealisedMetrics, sig: SignalState):
        C = StraddleVisualizer.S
        ax.set_facecolor(C.BG_CARD)
        ax.axis("off")

        score_val = snap.dealer_score
        if score_val is None:
            score_col, dealer_lbl, score_str = C.GREY, "N/A", "N/A"
        else:
            score_col  = C.GREEN if score_val > 15 else (C.RED if score_val < -15 else C.GOLD)
            dealer_lbl = "BULL" if score_val > 15 else ("BEAR" if score_val < -15 else "NEUT")
            score_str  = f"{score_val:+.0f}"

        theta_usd = sig.theta_per_hour_usd * sig.hours_to_expiry
        vrp_str   = f"{rm.vrp_mean:.2f}×" if not math.isnan(rm.vrp_mean) else "N/A"

        kpis = [
            (f"${snap.straddle_index:,.0f}",     "SPOT",         C.WHITE),
            (f"{snap.straddle_iv:.1f}%",         "IV",           C.CYAN),
            (f"{sig.remaining_edge_pct:+.2f}%",  "EDGE",         C.edge_color(sig.remaining_edge_pct)),
            (f"{sig.exhaustion_ratio:.0f}%",     "EXHAUSTION",   C.exhaustion_color(sig.exhaustion_ratio)),
            (rm.dvol_momentum.upper(),           "DVOL",         C.dvol_color(rm.dvol_momentum)),
            (vrp_str,                            "VRP",          C.vrp_color(rm.vrp_mean)),
            (f"${theta_usd:,.0f}",               "θ-TO-EXP",     C.GOLD),
            (snap.spot_momentum_label.upper(),   "MOMENTUM",     C.momentum_color(snap.spot_momentum_label)),
            (f"{score_str} [{dealer_lbl}]",      "DEALER SCORE", score_col),
        ]

        n   = len(kpis)
        sep = 1.0 / n
        for i, (val, lbl, col) in enumerate(kpis):
            cx = (i + 0.5) * sep
            ax.text(cx, 0.68, val, ha="center", va="center", fontsize=12.5,
                    weight="bold", color=col, transform=ax.transAxes)
            ax.text(cx, 0.16, lbl, ha="center", va="center", fontsize=7,
                    color=C.GREY, transform=ax.transAxes, fontfamily="monospace")
            if i < n - 1:
                ax.axvline((i + 1) * sep, color=C.GREY, lw=0.5, alpha=0.35)

    # ── Panel 1: Spot price vs priced-in move ───────────────────────
    @staticmethod
    def _draw_spot_price_panel(ax, snap: MarketSnapshot, rm: RealisedMetrics):
        C = StraddleVisualizer.S
        ax.set_facecolor(C.BG_PANEL)
        ax.set_title("1 · Spot Price vs Priced-In Move", fontsize=11, weight="bold", pad=6)

        if not rm.price_path or rm.anchor_price <= 0:
            ax.text(0.5, 0.5, "No intraday price data available", ha="center", va="center",
                    color=C.GREY, transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            return

        times  = [ist(t) for t, _ in rm.price_path]
        prices = [p for _, p in rm.price_path]

        anchor_price = rm.anchor_price
        implied_band = rm.starting_mv_price  # ≈ combined call+put breakeven width

        signed_move_pct = ((rm.current_price - rm.anchor_price) / rm.anchor_price * 100) if rm.anchor_price > 0 else 0.0
        move_col = C.GREEN if signed_move_pct >= 0 else C.RED

        if implied_band > 0:
            upper, lower = anchor_price + implied_band, anchor_price - implied_band
            ax.axhspan(lower, upper, color=C.CYAN, alpha=0.08,
                       label=f"Priced-in range (±${implied_band:,.0f})")
            mid_x = times[len(times) // 2]
            mid_y = (upper + lower) / 2
            ax.text(mid_x, mid_y, f"±${implied_band:,.0f}", color=C.CYAN,
                    ha="center", va="center", fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=C.BG_PANEL,
                              edgecolor=C.CYAN, alpha=0.65))
        else:
            upper = lower = anchor_price

        ax.axhline(anchor_price, color=C.WHITE, ls=":", lw=1.2, alpha=0.6,
                   label=f"Anchor ${anchor_price:,.0f}")
        ax.annotate(f"17:30 IST ${anchor_price:,.0f}", (times[0], anchor_price), xytext=(8, 10),
                    textcoords="offset points", fontsize=7.5, color=C.WHITE,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=C.BG_CARD, alpha=0.75),
                    ha="left", va="bottom")

        ax.plot(times, prices, color=C.ACCENT, lw=1.8, zorder=5, label="Spot")
        ax.fill_between(times, prices, anchor_price, color=C.ACCENT, alpha=0.08, zorder=1)

        if rm.window_high and rm.window_low:
            ax.axhline(rm.window_high, color=C.GOLD, ls="--", lw=0.8, alpha=0.45)
            ax.axhline(rm.window_low,  color=C.GOLD, ls="--", lw=0.8, alpha=0.45)

        signed_move_pct = ((rm.current_price - rm.anchor_price) / rm.anchor_price * 100) if rm.anchor_price > 0 else 0.0
        move_col = C.GREEN if signed_move_pct >= 0 else C.RED
        move_label = f"{signed_move_pct:+.2f}% from 17:30 IST"
        ax.annotate(move_label, (times[-1], prices[-1]), xytext=(8, 10), textcoords="offset points",
                    fontsize=8.2, weight="bold", color=move_col,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=C.BG_CARD, edgecolor=move_col, alpha=0.9))

        ax.annotate(f"{signed_move_pct:+.2f}% from 17:30 IST", (times[-1], prices[-1]), xytext=(8, 10),
                    textcoords="offset points", fontsize=8.2, weight="bold", color=move_col,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=C.BG_CARD, edgecolor=move_col, alpha=0.9))
        ax.scatter([times[-1]], [prices[-1]], color=C.WHITE, s=70, edgecolors=C.ACCENT,
                   lw=1.5, zorder=10, label=f"Now ${rm.current_price:,.0f}")

        if implied_band > 0:
            outside = rm.current_price > upper or rm.current_price < lower
            tag, tag_col = ("OUTSIDE priced range", C.RED) if outside else ("Inside priced range", C.GREEN)
            ax.text(0.02, 0.05, tag, transform=ax.transAxes, fontsize=8.5, weight="bold",
                    color=tag_col, bbox=dict(boxstyle="round,pad=0.3", facecolor=C.BG_CARD, alpha=0.85))

        summary = (f"Move {signed_move_pct:+.2f}% | Anchor ${rm.anchor_price:,.0f} | Now ${rm.current_price:,.0f}"
                   f" | Range ±${implied_band:,.0f}" if implied_band > 0 else
                   f"Move {signed_move_pct:+.2f}% | Anchor ${rm.anchor_price:,.0f} | Now ${rm.current_price:,.0f}")
        ax.text(0.02, 0.92, summary, transform=ax.transAxes, fontsize=8.1, weight="bold",
                color=C.WHITE, bbox=dict(boxstyle="round,pad=0.3", facecolor=C.BG_CARD, alpha=0.7))

        ax.set_ylabel("Price ($)", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.5)
        ax.grid(True, alpha=0.06)

    # ── Panel 2: Decision quadrant (edge vs exhaustion) ─────────────
    @staticmethod
    def _draw_decision_quadrant(ax, snap: MarketSnapshot, rm: RealisedMetrics, sig: SignalState):
        C = StraddleVisualizer.S
        ax.set_facecolor(C.BG_PANEL)

        ev, exh, ethr = sig.remaining_edge_pct, sig.exhaustion_ratio, Config.EXHAUSTION_THRESHOLD
        xleft, xright = min(-2.0, ev - 1.5), max(2.0, ev + 1.5)
        ytop = max(130, exh + 15)
        x0n  = (0 - xleft) / (xright - xleft)   # single source of truth for the vertical split

        ax.axhspan(ethr, ytop, xmin=x0n, xmax=1.0, color=C.GREEN,  alpha=0.12)
        ax.axhspan(ethr, ytop, xmin=0.0, xmax=x0n, color=C.PURPLE, alpha=0.12)
        ax.axhspan(0,    ethr, xmin=0.0, xmax=x0n, color=C.CYAN,   alpha=0.10)
        ax.axhspan(0,    ethr, xmin=x0n, xmax=1.0, color=C.GOLD,   alpha=0.10)

        for txt, xf, yf, col in [
            ("SHORT ZONE",  0.98, 0.98, C.GREEN),
            ("GAMMA BUY",   0.02, 0.98, C.PURPLE),
            ("VALUE BUY",   0.02, 0.02, C.CYAN),
            ("WAIT→SHORT",  0.98, 0.02, C.GOLD),
        ]:
            ax.text(txf if False else xf, yf, txt, transform=ax.transAxes,
                    ha="right" if xf > 0.5 else "left",
                    va="top" if yf > 0.5 else "bottom",
                    color=col, fontsize=8, weight="bold", alpha=0.85)

        ax.axvline(0, color=C.WHITE, ls=":", alpha=0.4)
        ax.axhline(ethr, color=C.WHITE, ls=":", alpha=0.4, label=f"Exhaustion thr {ethr:.0f}%")
        if sig.seasonal_std_pct > 0:
            ax.axvspan(-sig.seasonal_std_pct, sig.seasonal_std_pct, color=C.WHITE, alpha=0.04,
                       label="±1σ seasonal edge")

        # Regime history shown only as faint density context — not individually labeled
        for w in rm.regime_weeks:
            if w.mv_pct <= 0:
                continue
            w_exh  = (w.pct_range_hl / w.mv_pct) * 100
            w_edge = w.mv_pct - rm.seasonal_avg_pct
            ax.scatter([w_edge], [w_exh], color=C.GREY, s=22, alpha=0.30, zorder=3)

        sig_col = C.signal_color(sig)
        ax.scatter([ev], [exh], color=sig_col, s=280, edgecolors="white", lw=2, zorder=10)
        ax.annotate(f"  Edge {ev:+.2f}%\n  Exh {exh:.0f}%", (ev, exh), fontsize=8.5,
                    color=C.WHITE, weight="bold", xytext=(8, 8), textcoords="offset points")

        ax.set_title("2 · Decision: Edge vs Exhaustion", fontsize=11, weight="bold", pad=6)
        ax.set_xlabel("Market Edge vs Regime Avg (%)", fontsize=8)
        ax.set_ylabel("Exhaustion (%)", fontsize=8)
        ax.set_xlim(xleft, xright)
        ax.set_ylim(0, ytop)
        ax.grid(True, alpha=0.06)
        ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    # ── Panel 3: Realized vs implied $ by week (grouped bars) ───────
    @staticmethod
    def _draw_regime_bars(ax, rm: RealisedMetrics):
        C = StraddleVisualizer.S
        ax.set_facecolor(C.BG_PANEL)
        ax.set_title("3 · Realized vs Implied ($) by Week", fontsize=11, weight="bold", pad=6)

        rws = sorted(rm.regime_weeks, key=lambda w: w.expiry_utc)
        if not rws:
            ax.text(0.5, 0.5, "No regime weeks available", ha="center", va="center",
                    color=C.GREY, transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            return

        x        = np.arange(len(rws))
        realized = np.array([w.pct_move_c2c / 100 * w.start_price for w in rws])
        implied  = np.array([w.mv_price_usd for w in rws])
        xlbls    = [ist_str(w.expiry_utc, "%d%b") for w in rws]

        width = 0.38
        ax.bar(x - width / 2, realized, width, color=C.GOLD, alpha=0.85, label="Realized move $")
        ax.bar(x + width / 2, implied,  width, color=C.CYAN, alpha=0.85, label="Implied premium $")
        ax.axhline(float(np.mean(realized)), color=C.GOLD, ls="--", lw=1.0, alpha=0.6)
        ax.axhline(float(np.mean(implied)),  color=C.CYAN, ls="--", lw=1.0, alpha=0.6)

        step = 1 if len(xlbls) <= 12 else (2 if len(xlbls) <= 24 else 4)
        ax.set_xticks(x)
        ax.set_xticklabels([xlbls[i] if i % step == 0 else "" for i in range(len(xlbls))],
                            rotation=45, fontsize=6.5)

        win_rate = float(np.mean(implied > realized)) * 100
        ax.text(0.98, 0.95, f"Seller won {win_rate:.0f}% of weeks",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, weight="bold",
                color=C.GREEN if win_rate >= 50 else C.RED,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C.BG_CARD, alpha=0.85))

        ax.set_ylabel("$ (elapsed-aligned)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.06, axis="y")

    # ── Panel 4: VRP trend (single axis) ────────────────────────────
    @staticmethod
    def _draw_vrp_trend(ax, rm: RealisedMetrics):
        C = StraddleVisualizer.S
        ax.set_facecolor(C.BG_PANEL)
        ax.set_title("4 · Volatility Risk Premium Trend", fontsize=11, weight="bold", pad=6)

        rws = sorted([w for w in rm.regime_weeks if not math.isnan(w.vrp)], key=lambda w: w.expiry_utc)
        if not rws:
            ax.text(0.5, 0.5, "No VRP samples available", ha="center", va="center",
                    color=C.GREY, transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            return

        x     = np.arange(len(rws))
        vrps  = np.array([w.vrp for w in rws])
        xlbls = [ist_str(w.expiry_utc, "%d%b") for w in rws]
        colors = [C.GREEN if v >= 1.0 else C.RED for v in vrps]

        ax.bar(x, vrps, color=colors, alpha=0.8, width=0.6)
        ax.axhline(1.0, color=C.WHITE, ls=":", lw=1.0, alpha=0.5, label="Fair value (1.0×)")
        ax.axhline(rm.vrp_mean, color=C.CYAN, ls="--", lw=1.3, alpha=0.85,
                   label=f"Regime mean {rm.vrp_mean:.2f}×")
        if not math.isnan(rm.vrp_current):
            ax.axhline(rm.vrp_current, color=C.GOLD, ls="-", lw=1.3, alpha=0.85,
                       label=f"Today {rm.vrp_current:.2f}×")

        step = 1 if len(xlbls) <= 12 else (2 if len(xlbls) <= 24 else 4)
        ax.set_xticks(x)
        ax.set_xticklabels([xlbls[i] if i % step == 0 else "" for i in range(len(xlbls))],
                            rotation=45, fontsize=6.5)

        ax.set_ylabel("VRP (implied / realized)", fontsize=8)
        ax.set_ylim(0, max(float(vrps.max()) * 1.3, 2.0))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.06, axis="y")


# ── Telegram Reporter ─────────────────────────────────────────────
class TelegramReporter:
    """Delivers straddle visual report and caption to Telegram."""

    def __init__(self, session: aiohttp.ClientSession, bot_token: str = Config.TELEGRAM_BOT_TOKEN, chat_id: str = Config.TELEGRAM_CHAT_ID):
        self.session = session
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _split_message(self, text: str, max_len: int = 4000) -> List[str]:
        if len(text) <= max_len:
            return [text]

        chunks: List[str] = []
        lines = text.split("\n")
        current = ""

        for line in lines:
            if len(current) + len(line) + 1 > max_len:
                if current:
                    chunks.append(current)
                    current = line
                else:
                    while len(line) > max_len:
                        chunks.append(line[:max_len])
                        line = line[max_len:]
                    current = line
            else:
                current = f"{current}\n{line}" if current else line

        if current:
            chunks.append(current)

        return chunks

    async def send_message(self, text: str) -> bool:
        """Sends plain text chunks to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID is not configured")
            return False

        try:
            for chunk in self._split_message(text):
                payload = {
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                }
                async with self.session.post(
                    f"{self.base_url}/sendMessage",
                    data=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200 or not data.get("ok"):
                        err_txt = data.get("description", await resp.text())
                        logger.error(f"Telegram message failed: {resp.status} - {err_txt}")
                        return False
            logger.info("Telegram text sent successfully")
            return True
        except Exception as exc:
            logger.error(f"Telegram message error: {exc}", exc_info=True)
            return False

    async def send_report(self, img_buf: io.BytesIO, caption: str, sig: Optional[SignalState] = None) -> bool:
        """Sends the caption and chart image to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID is not configured")
            return False

        try:
            if caption:
                if not await self.send_message(caption):
                    return False

            await asyncio.sleep(0.4)
            img_buf.seek(0)
            form = aiohttp.FormData()
            form.add_field("chat_id", str(self.chat_id))
            form.add_field(
                "photo",
                img_buf.read(),
                filename="straddle_report.png",
                content_type="image/png",
            )

            async with self.session.post(
                f"{self.base_url}/sendPhoto",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200 or not data.get("ok"):
                    err_txt = data.get("description", await resp.text())
                    logger.error(f"Telegram image failed: {resp.status} - {err_txt}")
                    return False

            logger.info("Telegram chart image sent successfully")
            return True
        except Exception as exc:
            logger.error(f"Telegram report error: {exc}", exc_info=True)
            return False


# ── Discord Reporter ──────────────────────────────────────────────
class DiscordReporter:
    """Delivers straddle visual report and caption to Discord via webhook."""

    def __init__(self, session: aiohttp.ClientSession, webhook_url: str = Config.DISCORD_WEBHOOK_URL):
        self.session = session
        self.webhook_url = webhook_url

    def build_caption(
        self,
        snap:              MarketSnapshot,
        rm:                RealisedMetrics,
        sig:               SignalState,
        target_expiry_utc: datetime,
    ) -> str:
        anc_time = ist_str(rm.anchor_utc, "%H:%M")
        now_time = ist_str(snap.now_utc, "%H:%M")

        live_usd   = sig.market_price_usd
        regime_avg = sig.seasonal_avg_usd
        edge_word  = "RICH ↑" if sig.remaining_edge_pct > 0 else "CHEAP ↓"
        skew_str   = (
            f"{snap.skew_iv:+.2f}% (put−call IV)"
            if snap.skew_available else "N/A"
        )

        score_val  = snap.dealer_score
        score_str  = f"{score_val:+.0f}" if score_val is not None else "N/A"
        dealer_lbl = "🟢 BULL" if (score_val and score_val > 15) else ("🔴 BEAR" if (score_val and score_val < -15) else "🟡 NEUT")

        # Build regime table in monospace format
        rows = []
        mtm_sum = 0.0
        for w in sorted(rm.regime_weeks, key=lambda x: x.expiry_utc):
            exp_s  = ist_str(w.expiry_utc,  "%d-%b-%y %H:%M")
            dvol_s = f"{w.start_dvol:4.1f}" if not math.isnan(w.start_dvol) else " N/A"
            sym_s  = (w.mv_symbol or "N/A")
            if "SYNTH-" in sym_s:
                sym_s = sym_s.replace("SYNTH-", "") + "(S)"
            elif "-" in sym_s:
                parts = sym_s.split("-")
                sym_s = parts[2] if len(parts) >= 3 else sym_s
            r_usd = w.pct_move_c2c / 100 * w.start_price
            mtm_sym_s = str(w.mtm_strike)
            decay_pct = (w.mv_price_usd - w.mv_mtm_price_usd) / w.mv_price_usd * 100 if w.mv_price_usd > 0 else 0
            rows.append(
                f"{exp_s:<15} | {sym_s:<10} | {mtm_sym_s:<8} | ${w.mv_price_usd:>5.0f} | "
                f"${w.mv_mtm_price_usd:>5.0f} | {decay_pct:>4.0f}% | "
                f"{w.mtm_source:<3} | {dvol_s} | ${r_usd:>5.0f}"
            )
            mtm_sum += w.mv_mtm_price_usd

        avg_mtm = mtm_sum / len(rm.regime_weeks) if rm.regime_weeks else 0.0

        valid_dvols = [w.start_dvol for w in rm.regime_weeks if not math.isnan(w.start_dvol)]
        dvol_range  = f"{min(valid_dvols):.1f}-{max(valid_dvols):.1f}" if valid_dvols else "N/A"
        regime_table = ""
        if rows:
            footer_sep = "─" * 86
            curr_sym = (snap.delta_symbol or "BTC-ATM")
            if "SYNTH-" in curr_sym:
                curr_sym = curr_sym.replace("SYNTH-", "") + "(S)"
            elif "-" in curr_sym:
                parts = curr_sym.split("-")
                curr_sym = parts[2] if len(parts) >= 3 else curr_sym

            footer_current = f"Current straddle {curr_sym}'s: ${live_usd:,.0f}"
            footer_avg_mtm = f"Avg Historical MTM: ${avg_mtm:,.0f}"
            regime_table = (
                f"\n\n**📚 REGIME SAMPLES (DVOL: {dvol_range})**\n```text\n"
                f"{'Exp (IST)':<15} | {'Sym':<10} | {'MTM_Sym':<8} | {'Start$':>6} | {'MTM$':>6} | {'Dcy%':>5} | {'Src':<3} | {'DVOL':>5} | {'RV$':>6}\n"
                + "\n".join(rows)
                + f"\n{footer_sep}\n{footer_current}\n{footer_avg_mtm}"
                + "\n```"
            )

        consec_sfx = f" [x{sig.consecutive_count}]" if sig.consecutive_count > 1 else ""

        signal_icon = "🟢" if sig.conviction_label in ("VALUE BUY", "GAMMA BUY") else ("🔴" if "SHORT" in sig.conviction_label else "🟡")
        if sig.conviction_score >= 80:
            signal_icon = "🔥"

        return (
            f"**{signal_icon} VERDICT: {sig.conviction_label}** ({sig.conviction_score:.0f}/100){consec_sfx}\n"
            f"🛠️ **Strategy:** `{sig.strategy_suggestion}`\n"
            f"**📊 PRICING & EDGE**\n"
            f"• **Live Prem :** ${live_usd:,.0f}, symbol: {snap.delta_symbol}\n"
            f"• **Regime Avg:** ${regime_avg:,.0f} *({len(rm.regime_weeks)} wks matched)* [{anc_time} → {now_time}]\n"
            f"• **Regime Exh:** {sig.regime_exhaustion_avg:.0f}% *(hist. avg)* [{anc_time} → {now_time}]\n"
            f"• **Edge      :** {edge_word} {abs(sig.remaining_edge_pct):.2f}% (${abs(sig.remaining_edge_usd):,.0f})\n"
            f"• **Today's Exh:** {sig.exhaustion_ratio:.0f}% [{anc_time} → {now_time}] *(of initial ${rm.starting_mv_price:,.0f} [Full 24h])*\n\n"
            f"**🌊 MACRO & VOLATILITY**\n"
            f"• **Dealer Score:** {score_str} {dealer_lbl}\n"
            f"• **Spot Mom  :** {snap.spot_momentum_label} ({snap.spot_momentum_slope:+.0f} $/hr)\n"
            f"• **DVOL      :** {rm.current_dvol:.1f}% ({rm.dvol_momentum})\n"
            f"• **25d Skew  :** {skew_str}\n"
            f"• **VRP Mean  :** {rm.vrp_mean:.2f}x (n={rm.vrp_sample_count})"
            + regime_table
        )

    def _split_message(self, text: str, max_len: int = 1950) -> List[str]:
        if len(text) <= max_len:
            return [text]
        chunks = []
        lines = text.split("\n")
        current = ""
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # If adding this line exceeds max_len, flush current
            if len(current) + len(line) + 1 > max_len:
                if current:
                    if in_code_block:
                        current += "\n```"
                    chunks.append(current)
                    current = "```text\n" + line if in_code_block else line
                else:
                    while len(line) > max_len:
                        chunks.append(line[:max_len])
                        line = line[max_len:]
                    current = line
            else:
                current = f"{current}\n{line}" if current else line

        if current:
            chunks.append(current)
        return chunks

    async def send_report(self, img_buf: io.BytesIO, caption: str, sig: Optional[SignalState] = None) -> bool:
        """
        Sends 2 sequential messages:
        1. Discord rich embed message containing the complete caption & regime table.
        2. Visual chart image attachment as a standalone follow-up message.
        """
        if not self.webhook_url:
            logger.error("Discord webhook URL is not configured")
            return False

        # Determine embed color based on signal
        color = 0x3498DB  # Default Blue
        if sig:
            if "BUY" in sig.conviction_label:
                color = 0x2ECC71  # Green
            elif "SHORT" in sig.conviction_label or "SELL" in sig.conviction_label:
                color = 0xE74C3C  # Red
            elif "WAIT" in sig.conviction_label or "NEUTRAL" in sig.conviction_label:
                color = 0xF1C40F  # Gold/Yellow
        elif "BUY" in caption:
            color = 0x2ECC71
        elif "SHORT" in caption or "SELL" in caption:
            color = 0xE74C3C
        elif "WAIT" in caption:
            color = 0xF1C40F

        try:
            # ── Message 1: Rich Embed with full text caption ──
            # Discord embed description supports up to 4096 chars
            embed_chunks = self._split_message(caption, max_len=4000)
            for idx, chunk in enumerate(embed_chunks):
                embed = {
                    "description": chunk,
                    "color": color,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": f"BTC Straddle Engine v10 {'(Part ' + str(idx+1) + ')' if len(embed_chunks) > 1 else ''}"}
                }
                async with self.session.post(
                    self.webhook_url,
                    json={"embeds": [embed]},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status not in (200, 204):
                        err_txt = await resp.text()
                        logger.error(f"Discord embed message failed: {resp.status} - {err_txt}")
                        return False
                if idx < len(embed_chunks) - 1:
                    await asyncio.sleep(0.5)

            logger.info("Discord rich embed caption sent successfully")

            # ── Message 2: Chart image as separate message ──
            await asyncio.sleep(0.6)
            img_buf.seek(0)
            form = aiohttp.FormData()
            form.add_field(
                "file",
                img_buf.read(),
                filename="straddle_report.png",
                content_type="image/png",
            )
            async with self.session.post(
                self.webhook_url,
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status not in (200, 204):
                    err_txt = await resp.text()
                    logger.error(f"Discord chart image failed: {resp.status} - {err_txt}")
                    return False
                logger.info("Discord chart image sent successfully")

            return True
        except Exception as exc:
            logger.error(f"Discord report error: {exc}", exc_info=True)
            return False

    async def send_message(self, text: str) -> bool:
        """Sends a text message to Discord webhook."""
        if not self.webhook_url:
            return False
        try:
            chunks = self._split_message(text)
            for chunk in chunks:
                async with self.session.post(
                    self.webhook_url,
                    json={"content": chunk},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status not in (200, 204):
                        err_txt = await resp.text()
                        logger.error(f"Discord message failed: {resp.status} - {err_txt}")
                        return False
            logger.info("Discord message sent successfully")
            return True
        except Exception as exc:
            logger.error(f"Discord message error: {exc}")
            return False


# ── Main Orchestration ────────────────────────────────────────────
async def main():
    logger.info("=" * 60)
    logger.info("BTC Straddle Engine v10 (Standalone) — run start")
    logger.info("=" * 60)

    timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT_SECS * 3)
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout) as session:
        opts     = OptionsEngine(session)
        rm_eng   = RealisedMetricsEngine(session)
        caption_builder = DiscordReporter(session, Config.DISCORD_WEBHOOK_URL)
        sender = (
            TelegramReporter(session, Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
            if Config.OUTPUT_CHANNEL.lower() == "telegram"
            else DiscordReporter(session, Config.DISCORD_WEBHOOK_URL)
        )

        logger.info(f"Output channel: {Config.OUTPUT_CHANNEL.lower()}")

        # ── Step 1: Delta MV price + spot ──
        now_utc      = datetime.now(timezone.utc)
        delta_mv_raw = await opts.get_delta_mv_price()
        spot_price   = await opts.get_delta_spot_price()

        if not delta_mv_raw:
            target_expiry_utc = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
            if target_expiry_utc <= now_utc:
                target_expiry_utc += timedelta(days=1)
            delta_mv_raw = await opts.get_delta_synthetic_straddle(spot_price, target_expiry_utc)
        else:
            target_expiry_utc = delta_mv_raw["expiry"]

        hours_to_expiry = max(0.5, (target_expiry_utc - now_utc).total_seconds() / 3600)

        # ── Step 2: Resolve IV ──
        delta_iv = delta_mv_raw.get("iv", 0.0) if delta_mv_raw else 0.0
        if delta_iv <= 0:
            t_years   = hours_to_expiry / (365 * 24)
            price_pct = delta_mv_raw["price_pct"] if delta_mv_raw else 0.0
            delta_iv  = (price_pct / (0.8 * math.sqrt(t_years))) if t_years > 0 else 0.0
            logger.info(f"IV: Back-solved from premium: {delta_iv:.1f}%")
        else:
            logger.info(f"IV: Using native Delta Implied Vol: {delta_iv:.1f}%")

        # ── Step 3: Live DVOL ──
        live_dvol = await opts.get_live_dvol()
        logger.info(f"Live DVOL: {live_dvol:.1f}%")
        logger.info(
            f"Now      : {ist_str(now_utc)}\n"
            f"Anchor   : {ist_str(canonical_anchor(target_expiry_utc))}\n"
            f"Expiry   : {ist_str(target_expiry_utc)}\n"
            f"Remaining: {hours_to_expiry:.1f}h"
        )

        async def fetch_macro():
            if DeribitSynthesizer:
                try:
                    def run_macro():
                        synth = DeribitSynthesizer("BTC")
                        return synth.get_macro_environment()
                    return await asyncio.to_thread(run_macro)
                except Exception as e:
                    logger.error(f"Macro fetch failed: {e}")
            return None

        # ── Step 4: Parallel supplementary fetches ──
        results = await asyncio.gather(
            opts.get_25d_skew(target_expiry_utc, spot_price, delta_iv or live_dvol),
            opts.get_gamma_pin(target_expiry_utc, spot_price),
            rm_eng.compute(
                now_utc, target_expiry_utc, hours_to_expiry,
                current_dvol=live_dvol,
                current_symbol=delta_mv_raw["symbol"] if delta_mv_raw else None,
            ),
            opts.get_spot_momentum(),
            fetch_macro(),
            return_exceptions=True,
        )

        def safe_get(idx, default):
            res = results[idx]
            if isinstance(res, Exception):
                logger.error(f"Task {idx} failed: {res}")
                return default
            return res or default

        skew_res     = safe_get(0, {"available": False})
        pin_res      = safe_get(1, {"available": False})
        rm           = safe_get(2, None)
        momentum_res = safe_get(3, {"slope": 0.0, "label": "flat"})
        macro_res    = safe_get(4, None)

        # ── Step 5: Populate snapshot ──
        snap = MarketSnapshot(
            straddle_expiry     = target_expiry_utc,
            straddle_strike     = int(round(spot_price / 500) * 500),
            straddle_iv         = delta_iv,
            straddle_prem_pct   = delta_mv_raw["price_pct"] if delta_mv_raw else 0.0,
            straddle_prem_usd   = delta_mv_raw["price_usd"] if delta_mv_raw else 0.0,
            straddle_index      = spot_price,
            day_of_week         = now_utc.strftime("%A"),
            now_utc             = now_utc,
            delta_symbol        = delta_mv_raw["symbol"]    if delta_mv_raw else None,
            delta_price_usd     = delta_mv_raw["price_usd"] if delta_mv_raw else 0.0,
            delta_price_pct     = delta_mv_raw["price_pct"] if delta_mv_raw else 0.0,
            delta_expiry        = delta_mv_raw["expiry"]    if delta_mv_raw else None,
            delta_available     = delta_mv_raw is not None,
            skew_iv             = skew_res.get("skew_iv", 0.0),
            skew_vol_imbalance  = skew_res.get("skew_vol_imbalance", 0.0),
            skew_composite      = skew_res.get("skew_composite", 0.0),
            skew_available      = skew_res.get("available", False),
            gamma_pin_strike    = pin_res.get("pin_strike"),
            gamma_pin_oi_pct    = pin_res.get("oi_pct", 0.0),
            spot_momentum_slope = momentum_res.get("slope", 0.0),
            spot_momentum_label = momentum_res.get("label", "flat"),
            dealer_score        = macro_res.get("dealer_score") if macro_res else None,
        )

        logger.info(f"PRICING | Delta Premium: {snap.delta_price_pct:.3f}% (${snap.delta_price_usd:,.2f})")

        # ── Step 6: Signal ──
        sig = compute_signal(snap, rm, now_utc, target_expiry_utc, snap.delta_price_pct)

        # ── Step 7: Chart + caption ──
        img_buf = StraddleVisualizer.generate_report(snap, rm, sig)
        caption = caption_builder.build_caption(snap, rm, sig, target_expiry_utc)

        # ── Step 8: Send to configured output channel ──
        await sender.send_report(img_buf, caption, sig)

        # ── Step 9: Persist run history ──
        append_run_history(snap, rm, sig, target_expiry_utc)

        logger.info("=" * 60)
        logger.info("Run complete.")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

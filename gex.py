import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import pytz
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

TELEGRAM_BOT_TOKEN = ${{ secrets.TELEGRAM_BOT_TOKEN }}
TELEGRAM_CHAT_ID = ${{ secrets.TELEGRAM_CHAT_ID }}

# define in the code num_expiries
# define in the code: price_range (spot +/- price_range, used for the GEX table/strings)
# define in the code: chart_price_range (spot +/- chart_price_range, used only for the bar chart x-axis)
# define in line ~ pine_strikes = list(range(65000, 101000, 1000))

BASE_URL = "https://www.deribit.com/api/v2"
HEADERS = {"Accept": "application/json"}

INDIA_TZ = pytz.timezone('Asia/Kolkata')

# ==== Data Fetching ====
def get_instruments():
    """Fetch all BTC option instruments from Deribit."""
    url = f"{BASE_URL}/public/get_instruments?currency=BTC&kind=option&expired=false"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()['result']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching instruments: {e}")
        return None


def fetch_ticker(instrument_name, retries=3, backoff_factor=2):
    """Fetch the FULL ticker payload (greeks, OI, mark_iv, bid/ask/last/mark price) for a
    single instrument, with retries and exponential backoff. This is the ONLY place in the
    script that hits the /public/ticker endpoint - every other function reads from the
    shared ticker_cache built by fetch_all_tickers() below, instead of re-fetching."""
    url = f"{BASE_URL}/public/ticker?instrument_name={instrument_name}"
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.json().get('result', {})
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ticker for {instrument_name}: {e}. Attempt {i + 1}/{retries}")
            if i < retries - 1:
                time.sleep(backoff_factor * (i + 1))
            else:
                print(f"Final attempt failed for {instrument_name}. Returning empty result.")
    return {}


def fetch_all_tickers(instrument_names):
    """Fetch ticker data for a list of instrument names CONCURRENTLY and EXACTLY ONCE per
    instrument, returning a {instrument_name: ticker_dict} cache that every downstream
    calculation (GEX table, weekly GEX string, implied move, straddle premium) reads from."""
    unique_names = list(dict.fromkeys(instrument_names))  # de-dupe while preserving order
    total = len(unique_names)
    cache = {}
    print(f"Fetching ticker data for {total} unique instruments concurrently (single pass)...")
    sys.stdout.flush()

    # Max workers = 5 keeps requests well below the 20 req/s rate limit.
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_ticker, name): name for name in unique_names}
        for idx, future in enumerate(as_completed(futures)):
            name = futures[future]
            cache[name] = future.result()
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"Progress: {idx + 1}/{total} tickers fetched.")
                sys.stdout.flush()
    return cache


def extract_gamma_oi_iv(ticker_data):
    """Pull gamma, open interest and mark IV out of an already-fetched ticker payload."""
    gamma = ticker_data.get('greeks', {}).get('gamma', 0.0)
    oi = ticker_data.get('open_interest', 0)
    mark_iv = ticker_data.get('mark_iv', 0.0)
    return gamma, oi, mark_iv


def extract_mid_price(ticker_data):
    """Return a usable price from an already-fetched ticker payload: best bid/ask mid first,
    falling back to Deribit's mark_price (always populated, unlike last_price on illiquid
    short-dated strikes), then last_price, then 0.0 as a last resort."""
    bid = ticker_data.get('best_bid_price')
    ask = ticker_data.get('best_ask_price')
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2

    mark = ticker_data.get('mark_price')
    if mark and mark > 0:
        return mark

    last = ticker_data.get('last_price')
    if last and last > 0:
        return last

    return 0.0


def get_current_btc_price():
    """Fetch the current BTC index price from Deribit."""
    url = f"{BASE_URL}/public/get_index_price?index_name=btc_usd"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()['result']['index_price']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching BTC price: {e}")
        return None


def get_target_expiries_and_instruments(all_instruments, current_price, price_range=20000):
    """Filter instruments for the first sheet using the next 6 expiries."""
    all_ts = sorted(list(set(inst['expiration_timestamp'] for inst in all_instruments)))
    future_ts = [
        ts for ts in all_ts
        if datetime.fromtimestamp(ts // 1000, timezone.utc).date() >= datetime.now(timezone.utc).date()
    ]
    next_ts = sorted(future_ts)[:6]

    lower_bound = int((current_price - price_range) // 1000 * 1000)
    upper_bound = int((current_price + price_range) // 1000 * 1000)
    table_strikes = set(range(lower_bound, upper_bound + 1000, 1000))
    string_strikes = set(range(50000, 101000, 1000))
    union_strikes = table_strikes | string_strikes

    filtered_instruments = [
        inst for inst in all_instruments
        if inst['expiration_timestamp'] in next_ts and int(inst['strike']) in union_strikes
    ]

    return filtered_instruments, next_ts, table_strikes, string_strikes


def get_weekly_expiry_cutoff_date(now=None):
    """Return the nearest Friday date on or after the current UTC date."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    days_until_friday = (4 - today.weekday()) % 7
    return today + timedelta(days=days_until_friday)


def get_this_week_expiries(all_instruments):
    """Get expiry timestamps up to the upcoming weekly expiry date for the second sheet string."""
    all_ts = sorted(list(set(inst['expiration_timestamp'] for inst in all_instruments)))
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    future_ts = [
        ts for ts in all_ts
        if datetime.fromtimestamp(ts // 1000, timezone.utc).date() >= today
    ]

    if not future_ts:
        return []

    weekly_cutoff = get_weekly_expiry_cutoff_date(now_utc)
    expiries_this_week = [
        ts for ts in future_ts
        if today <= datetime.fromtimestamp(ts // 1000, timezone.utc).date() <= weekly_cutoff
    ]

    if expiries_this_week:
        return sorted(expiries_this_week)

    return [future_ts[0]]


def get_weekly_gex_instruments(all_instruments, current_price, price_range=20000):
    """Filter instruments for the weekly-expiry GEX string and implied move."""
    weekly_expiry_timestamps = get_this_week_expiries(all_instruments)
    if not weekly_expiry_timestamps:
        return [], [], set()

    lower_bound = int((current_price - price_range) // 1000 * 1000)
    upper_bound = int((current_price + price_range) // 1000 * 1000)
    string_strikes = set(range(50000, 101000, 1000))
    union_strikes = set(range(lower_bound, upper_bound + 1000, 1000)) | string_strikes

    filtered_instruments = [
        inst for inst in all_instruments
        if inst['expiration_timestamp'] in weekly_expiry_timestamps and int(inst['strike']) in union_strikes
    ]

    return filtered_instruments, weekly_expiry_timestamps, string_strikes


def get_atm_instruments(all_instruments, current_price, expiry_ts):
    """Return the ATM (call & put) instrument dicts for a given expiry timestamp.
    Pure filtering over the already-fetched instrument list - no API call."""
    if expiry_ts is None:
        return []

    expiry_instruments = [inst for inst in all_instruments if inst['expiration_timestamp'] == expiry_ts]
    if not expiry_instruments:
        return []

    strikes = sorted({int(inst['strike']) for inst in expiry_instruments})
    if not strikes:
        return []

    atm_strike = min(strikes, key=lambda s: abs(s - current_price))
    return [inst for inst in expiry_instruments if int(inst['strike']) == atm_strike]


def get_weekly_implied_move_range(all_instruments, current_price, weekly_expiry_timestamps, ticker_cache):
    """Calculate a 1SD implied move range for the latest weekly expiry using ATM IV,
    reading mark_iv from the shared ticker_cache instead of re-fetching."""
    if not weekly_expiry_timestamps:
        return "N/A"

    target_expiry_ts = max(weekly_expiry_timestamps)
    atm_instruments = get_atm_instruments(all_instruments, current_price, target_expiry_ts)
    if not atm_instruments:
        return "N/A"

    iv_values = []
    for inst in atm_instruments:
        ticker_data = ticker_cache.get(inst['instrument_name'], {})
        _, _, mark_iv = extract_gamma_oi_iv(ticker_data)
        if mark_iv:
            iv_values.append(float(mark_iv))

    if not iv_values:
        return "N/A"

    avg_atm_iv = sum(iv_values) / len(iv_values)
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    dte = max((target_expiry_ts - now_ms) / (1000 * 60 * 60 * 24), 1 / 365)
    implied_move = current_price * (avg_atm_iv / 100) * np.sqrt(dte / 365)
    lower_bound = current_price - implied_move
    upper_bound = current_price + implied_move

    return f"{int(lower_bound):,} to {int(upper_bound):,}"


def get_next_expiry_timestamp(all_instruments, min_hours=0):
    """Return the next expiry timestamp after now, skipping expiries under min_hours if requested."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    all_ts = sorted({inst['expiration_timestamp'] for inst in all_instruments if inst['expiration_timestamp'] > now_ms})
    for ts in all_ts:
        hours_to_expiry = (ts - now_ms) / (1000 * 60 * 60)
        if hours_to_expiry >= min_hours:
            return ts
    return None


def get_atm_straddle_premium(all_instruments, current_price, expiry_timestamp, ticker_cache):
    """Return the ATM straddle premium (in BTC) for a specific expiry timestamp, reading
    prices from the shared ticker_cache instead of re-fetching. Falls back through
    bid/ask mid -> mark_price -> last_price (see extract_mid_price) so illiquid, no-trade
    instruments don't silently come back as 0."""
    if expiry_timestamp is None:
        return None

    atm_instruments = get_atm_instruments(all_instruments, current_price, expiry_timestamp)
    call_inst = next((inst for inst in atm_instruments if inst['option_type'] == 'call'), None)
    put_inst = next((inst for inst in atm_instruments if inst['option_type'] == 'put'), None)

    if not call_inst or not put_inst:
        return None

    call_ticker = ticker_cache.get(call_inst['instrument_name'], {})
    put_ticker = ticker_cache.get(put_inst['instrument_name'], {})
    call_price = extract_mid_price(call_ticker)
    put_price = extract_mid_price(put_ticker)

    if call_price <= 0:
        print(f"WARNING: No usable price found for {call_inst['instrument_name']} (straddle call leg).")
    if put_price <= 0:
        print(f"WARNING: No usable price found for {put_inst['instrument_name']} (straddle put leg).")

    return call_price + put_price


def calculate_gex_data(instruments_to_process, ticker_cache):
    """Compute GEX per expiry/strike/option-type from an already-fetched ticker_cache.
    No network calls happen here - this is pure processing of data fetched once up front."""
    gex_data = {}
    for inst in instruments_to_process:
        name = inst['instrument_name']
        ticker_data = ticker_cache.get(name, {})
        gamma, oi, _ = extract_gamma_oi_iv(ticker_data)

        expiry_datetime = datetime.fromtimestamp(inst['expiration_timestamp'] // 1000, timezone.utc)
        expiry_label = expiry_datetime.strftime('%Y-%m-%d')
        strike = int(inst['strike'])
        option_type = inst['option_type']

        gex = oi * gamma * 1000

        if expiry_label not in gex_data:
            gex_data[expiry_label] = {}
        if strike not in gex_data[expiry_label]:
            gex_data[expiry_label][strike] = {'call': 0, 'put': 0}
        gex_data[expiry_label][strike][option_type] = gex

    return gex_data


def build_expiry_label(expiry_timestamps):
    """Create a readable date label listing each expiry date individually."""
    if not expiry_timestamps:
        return "N/A"

    expiry_dates = sorted({
        datetime.fromtimestamp(ts // 1000, timezone.utc).date()
        for ts in expiry_timestamps
    })

    return ", ".join(date.strftime('%d %b %Y') for date in expiry_dates)


def build_total_gex_by_strike(gex_data, strikes):
    """Calculate total net GEX per strike across all expiries in gex_data, restricted to `strikes`."""
    strikes_sorted = sorted(strikes)
    expiries_sorted = sorted(gex_data.keys())
    df = pd.DataFrame(index=strikes_sorted, columns=expiries_sorted).apply(pd.to_numeric).fillna(0)
    for exp in expiries_sorted:
        for strike in strikes_sorted:
            call_gex = gex_data.get(exp, {}).get(strike, {}).get('call', 0)
            put_gex = gex_data.get(exp, {}).get(strike, {}).get('put', 0)
            df.loc[strike, exp] = call_gex - put_gex
    total_gex = df.sum(axis=1)
    return total_gex


def generate_gex_bar_chart(total_gex_by_strike, current_price, expiry_date_str):
    """Generate a PNG bar chart of total GEX by strike with spot price annotation.
    The x-axis is driven entirely by the strikes passed in (caller controls the window,
    e.g. spot +/- 10k), so no extra clipping logic is needed here."""
    fig, ax = plt.subplots(figsize=(14, 7))
    strikes = [int(x) for x in total_gex_by_strike.index]
    values = np.array(total_gex_by_strike.values, dtype=float)

    bar_colors = ['tab:blue'] * len(strikes)
    if len(values) > 0:
        positive_mask = values > 0
        negative_mask = values < 0
        if positive_mask.any():
            highest_pos_index = int(np.argmax(values * positive_mask))
            bar_colors[highest_pos_index] = 'tab:green'
        if negative_mask.any():
            highest_neg_index = int(np.argmin(values * negative_mask))
            bar_colors[highest_neg_index] = 'tab:red'

    bar_width = 600
    ax.bar(strikes, values, color=bar_colors, width=bar_width)
    ax.axvline(current_price, color='red', linestyle='--', linewidth=2, label=f"Spot {int(current_price):,}")
    ax.set_title(f"Total GEX by Strike till {expiry_date_str}")
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Net GEX')

    if len(strikes) > 0:
        ax.set_xlim(min(strikes) - 500, max(strikes) + 500)

    if len(strikes) > 20:
        tick_step = max(1, len(strikes) // 20)
        ax.set_xticks(strikes[::tick_step])
    else:
        ax.set_xticks(strikes)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def post_chart_to_telegram(bot_token, chat_id, image_bytes, content):
    """Post a generated chart image to Telegram via the bot API."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        data = {
            'chat_id': chat_id,
            'caption': content,
            'parse_mode': 'HTML'
        }
        files = {'photo': ('gex_chart.png', image_bytes, 'image/png')}
        resp = requests.post(url, data=data, files=files, timeout=30)
        resp.raise_for_status()
        print("SUCCESS: Posted GEX chart image to Telegram.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to post GEX chart to Telegram: {e}")


# ==== Main Runner ====
def main_gex_monitor():
    current_price = get_current_btc_price()
    all_instruments = get_instruments()
    if not current_price or not all_instruments:
        print("Data fetch failed. Skipping run.")
        return
    print(f"Fetched BTC price: {current_price}")

    price_range = 20000          # spot +/- this range drives the GEX table / GEX strings
    chart_price_range = 10000    # spot +/- this range drives ONLY the bar chart x-axis

    # --- Pure filtering over already-fetched instrument list (no API calls) ---
    filtered_inst, _, table_strikes, string_strikes = get_target_expiries_and_instruments(
        all_instruments, current_price, price_range=price_range
    )

    weekly_instruments, weekly_expiry_timestamps, weekly_string_strikes = get_weekly_gex_instruments(
        all_instruments, current_price, price_range=price_range
    )
    weekly_string_strikes_sorted = sorted(list(weekly_string_strikes))

    if not filtered_inst and not weekly_instruments:
        print("No instruments in range.")
        return

    # ATM legs needed for the weekly implied-move calc and the ATM straddle premium.
    # These almost always overlap with filtered_inst/weekly_instruments already, but we
    # add them explicitly (and de-dupe below) so nothing is ever fetched twice OR missed.
    weekly_target_ts = max(weekly_expiry_timestamps) if weekly_expiry_timestamps else None
    weekly_atm_instruments = get_atm_instruments(all_instruments, current_price, weekly_target_ts)

    next_expiry_ts = get_next_expiry_timestamp(all_instruments, min_hours=3)
    straddle_atm_instruments = get_atm_instruments(all_instruments, current_price, next_expiry_ts)

    # --- Build ONE de-duplicated instrument set and fetch each ticker exactly once ---
    combined_gex_map = {inst['instrument_name']: inst for inst in filtered_inst}
    for inst in weekly_instruments:
        combined_gex_map.setdefault(inst['instrument_name'], inst)
    combined_gex_instruments = list(combined_gex_map.values())

    all_needed_map = dict(combined_gex_map)
    for inst in weekly_atm_instruments + straddle_atm_instruments:
        all_needed_map.setdefault(inst['instrument_name'], inst)

    ticker_cache = fetch_all_tickers([inst['instrument_name'] for inst in all_needed_map.values()])

    # --- Everything below is pure processing of the single ticker_cache fetched above ---
    print("Processing filtered strikes...")
    gex_data = calculate_gex_data(combined_gex_instruments, ticker_cache)

    all_expiries_labels = sorted(list(gex_data.keys()))

    weekly_expiry_labels = sorted({
        datetime.fromtimestamp(ts // 1000, timezone.utc).strftime('%Y-%m-%d')
        for ts in weekly_expiry_timestamps
    })

    weekly_gex_subset = {
        exp: gex_data.get(exp, {})
        for exp in weekly_expiry_labels
        if exp in gex_data
    }

    # --- GEX string for all expiries expiring this week ---
    print("Generating GEX string for all expiries expiring this week...")
    df_gex_this_week = pd.DataFrame(index=sorted(weekly_gex_subset.keys()), columns=weekly_string_strikes_sorted).apply(pd.to_numeric).fillna(0)
    for exp in sorted(weekly_gex_subset.keys()):
        for strike in weekly_string_strikes_sorted:
            call_gex = weekly_gex_subset.get(exp, {}).get(strike, {}).get('call', 0)
            put_gex = weekly_gex_subset.get(exp, {}).get(strike, {}).get('put', 0)
            df_gex_this_week.loc[exp, strike] = call_gex - put_gex

    total_gex_series_this_week = df_gex_this_week.sum(axis=0)
    gex_values_list_this_week = [str(int(val)) for val in total_gex_series_this_week]
    this_week_final_str = ",".join(gex_values_list_this_week)

    # --- Next-expiry ATM straddle premium (fixed 0-value bug + expiry label for A7) ---
    next_expiry_label = None
    if next_expiry_ts:
        next_expiry_label = datetime.fromtimestamp(next_expiry_ts // 1000, timezone.utc).strftime('%Y-%m-%d')

    straddle_premium_btc = get_atm_straddle_premium(all_instruments, current_price, next_expiry_ts, ticker_cache)
    straddle_premium_usd = None
    if straddle_premium_btc is None or straddle_premium_btc <= 0:
        straddle_premium_str = 'N/A'
    else:
        straddle_premium_usd = straddle_premium_btc * current_price
        straddle_premium_str = f"{straddle_premium_btc:.4f} BTC (${straddle_premium_usd:,.0f})"

    straddle_expiry_label = next_expiry_label if next_expiry_label else "N/A"

    # --- GEX string for the next expiry only ---
    print("Generating GEX string for next expiry...")
    if next_expiry_label and next_expiry_label in gex_data:
        next_expiry_to_use = next_expiry_label
    elif all_expiries_labels:
        next_expiry_to_use = all_expiries_labels[0]
    else:
        next_expiry_to_use = None

    if next_expiry_to_use:
        df_gex_1_exp = pd.DataFrame(index=[next_expiry_to_use], columns=weekly_string_strikes_sorted).apply(pd.to_numeric).fillna(0)
        for strike in weekly_string_strikes_sorted:
            call_gex = gex_data.get(next_expiry_to_use, {}).get(strike, {}).get('call', 0)
            put_gex = gex_data.get(next_expiry_to_use, {}).get(strike, {}).get('put', 0)
            df_gex_1_exp.loc[next_expiry_to_use, strike] = call_gex - put_gex
        total_gex_series_1_exp = df_gex_1_exp.sum(axis=0)
        gex_values_list_1_exp = [str(int(val)) for val in total_gex_series_1_exp]
        next_exp_final_str = ",".join(gex_values_list_1_exp)
    else:
        next_exp_final_str = "N/A"

    # --- Generate chart (x-axis now spot +/- chart_price_range, not the full 50k-100k range) ---
    try:
        chart_lower = int((current_price - chart_price_range) // 1000 * 1000)
        chart_upper = int((current_price + chart_price_range) // 1000 * 1000)
        chart_strikes = set(range(chart_lower, chart_upper + 1000, 1000))

        chart_total_gex_by_strike = build_total_gex_by_strike(weekly_gex_subset, chart_strikes)
        expiry_date_str = max(
            datetime.strptime(exp, '%Y-%m-%d').date() for exp in weekly_expiry_labels
        ).strftime('%d-%m-%Y') if weekly_expiry_labels else 'N/A'
        chart_bytes = generate_gex_bar_chart(chart_total_gex_by_strike, current_price, expiry_date_str)
        telegram_content = f"GEX bar chart for BTC strikes around {int(current_price):,} until {expiry_date_str}."
        post_chart_to_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, chart_bytes, telegram_content)
    except Exception as e:
        print(f"ERROR: Failed to create or post weekly GEX chart to Discord: {e}")

    this_week_metric_label = build_expiry_label(weekly_expiry_timestamps)
    weekly_implied_move_str = get_weekly_implied_move_range(all_instruments, current_price, weekly_expiry_timestamps, ticker_cache)

    print("GEX Monitor update complete.")


if __name__ == "__main__":
    try:
        print(f"Script started at {datetime.now(INDIA_TZ).strftime('%Y-%m-%d %H:%M:%S')} IST")
        main_gex_monitor()
    except Exception as e:
        print(f"Fatal error: {traceback.format_exc()}")
        print(f"ERROR: GEX Table Script Error: {e}")

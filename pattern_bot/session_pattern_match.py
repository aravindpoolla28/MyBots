from __future__ import annotations

import argparse
from typing import Dict, Iterable, Tuple

import pandas as pd

SESSION_WINDOWS = {
    "asia": (0, 7),
    "london": (8, 15),
    "ny": (16, 23),
}


def classify_move(change_pct: float, flat_tolerance: float = 0.01) -> str:
    """Classify a price move into up/down/flat buckets."""
    if abs(change_pct) <= flat_tolerance:
        return "flat"
    return "up" if change_pct > 0 else "down"


def _session_move_for_day(day_frame: pd.DataFrame, session_name: str, session_hours: Dict[str, Tuple[int, int]]) -> Tuple[float, str]:
    start_hour, end_hour = session_hours[session_name]
    session_rows = day_frame[
        (day_frame["timestamp"].dt.hour >= start_hour) & (day_frame["timestamp"].dt.hour <= end_hour)
    ]
    if session_rows.empty:
        return 0.0, "unknown"

    session_open = float(session_rows.iloc[0]["open"])
    session_close = float(session_rows.iloc[-1]["close"])
    move = (session_close - session_open) / session_open if session_open else 0.0
    return move, classify_move(move)


def build_session_history(df: pd.DataFrame, flat_tolerance: float = 0.02, session_hours: Dict[str, Tuple[int, int]] | None = None) -> pd.DataFrame:
    """Convert hourly OHLC data into daily session classification rows."""
    if session_hours is None:
        session_hours = SESSION_WINDOWS

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if not {"timestamp", "open", "close"}.issubset(frame.columns):
        raise ValueError("DataFrame must include timestamp, open, and close columns.")

    if not frame.empty:
        latest_day = frame["timestamp"].dt.date.max()
        latest_day_rows = frame[frame["timestamp"].dt.date == latest_day]
        if len(latest_day_rows) < 24:
            frame = frame[frame["timestamp"].dt.date < latest_day].copy()

    rows = []
    prev_close = None

    for day, day_frame in frame.groupby(frame["timestamp"].dt.date):
        day_frame = day_frame.sort_values("timestamp").reset_index(drop=True)
        if day_frame.empty:
            continue

        day_open = float(day_frame.iloc[0]["open"])
        if prev_close is not None and prev_close != 0:
            open_move = (day_open - prev_close) / prev_close
            open_label = classify_move(open_move, flat_tolerance)
        else:
            open_label = "unknown"

        session_snapshot = {}
        for session_name in ["asia", "london", "ny"]:
            move, label = _session_move_for_day(day_frame, session_name, session_hours)
            session_snapshot[session_name] = {"move": move, "label": label}

        if "asia" not in session_snapshot or "london" not in session_snapshot:
            prev_close = float(day_frame.iloc[-1]["close"])
            continue

        ny_label = session_snapshot.get("ny", {}).get("label", "unknown")
        record = {
            "date": day,
            "open_label": open_label,
            "asia_label": session_snapshot["asia"]["label"],
            "london_label": session_snapshot["london"]["label"],
            "ny_label": ny_label,
            "pattern": (open_label, session_snapshot["asia"]["label"], session_snapshot["london"]["label"]),
        }
        rows.append(record)

        prev_close = float(day_frame.iloc[-1]["close"])

    return pd.DataFrame(rows)


def _normalize_label_for_bucket(label: str) -> str:
    if label == "flat":
        return "neutral"
    return label


def _pattern_variants(open_label: str, asia_label: str, london_label: str):
    """Allow wider bucket matching so flat sessions can match a broader neutral family."""
    variants = set()
    labels = [open_label, asia_label, london_label]

    def expand(label: str):
        if label == "flat":
            return ["neutral", "up", "down"]
        if label == "unknown":
            return ["unknown", "neutral", "up", "down"]
        return [label]

    for open_variant in expand(labels[0]):
        for asia_variant in expand(labels[1]):
            for london_variant in expand(labels[2]):
                variants.add((open_variant, asia_variant, london_variant))

    return variants


def build_pattern_summary(history: pd.DataFrame, open_label: str, asia_label: str, london_label: str, flat_tolerance: float = 0.02):
    """Construct the structured pattern-match summary data used for both console output and string rendering."""
    if history.empty:
        return {
            "status": "no_data",
            "message": "No historical data available for this pattern match.",
        }

    exact_mask = (
        (history["open_label"] == open_label)
        & (history["asia_label"] == asia_label)
        & (history["london_label"] == london_label)
    )
    exact_matches = history[exact_mask].copy()

    if exact_matches.empty:
        history_variant = history.copy()
        for col in ["open_label", "asia_label", "london_label"]:
            history_variant[col] = history_variant[col].map(lambda value: _normalize_label_for_bucket(value))

        selectors = []
        for pattern in _pattern_variants(open_label, asia_label, london_label):
            selectors.append(
                (history_variant["open_label"] == _normalize_label_for_bucket(pattern[0]))
                & (history_variant["asia_label"] == _normalize_label_for_bucket(pattern[1]))
                & (history_variant["london_label"] == _normalize_label_for_bucket(pattern[2]))
            )

        mask = selectors[0]
        for condition in selectors[1:]:
            mask = mask | condition

        matches = history[mask].copy()
        match_mode = "broader bucket"
    else:
        matches = exact_matches
        match_mode = "exact"

    if matches.empty:
        return {
            "status": "no_match",
            "pattern": (open_label, asia_label, london_label),
            "historical_days": len(history),
            "message": (
                f"No NY-session historical matches found for open={open_label}, Asia={asia_label}, London={london_label} "
                f"based on {len(history)} historical days. Try widening the flat tolerance or reviewing a broader lookback window."
            ),
        }

    ny_counts = matches["ny_label"].fillna("unknown").value_counts().to_dict()
    total_matches = int(len(matches))
    up_count = int(ny_counts.get("up", 0))
    down_count = int(ny_counts.get("down", 0))
    flat_count = int(ny_counts.get("flat", 0))
    unknown_count = int(ny_counts.get("unknown", 0))
    all_dates = pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d").tolist()
    matched_dates = all_dates[:10]
    extra_dates = max(0, len(all_dates) - len(matched_dates))
    dates_text = ", ".join(matched_dates)
    if extra_dates:
        dates_text += f", ... (+{extra_dates} more)"

    best_label = max(
        ["up", "down", "flat"],
        key=lambda label: ny_counts.get(label, 0),
    )

    return {
        "status": "match",
        "best_label": best_label,
        "match_mode": match_mode,
        "pattern": (open_label, asia_label, london_label),
        "historical_days": len(history),
        "total_matches": total_matches,
        "dates_text": dates_text,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "unknown_count": unknown_count,
    }


def summarize_pattern(history: pd.DataFrame, open_label: str, asia_label: str, london_label: str, flat_tolerance: float = 0.02) -> str:
    """Return a probabilistic summary for a pattern match on NY session behavior."""
    result = build_pattern_summary(history, open_label, asia_label, london_label, flat_tolerance)
    if result["status"] == "no_data":
        return result["message"]
    if result["status"] == "no_match":
        return result["message"]

    return (
        f"Today's NY-session price action is most likely {result['best_label']} based on {result['historical_days']} historical days. "
        f"This {result['match_mode']} pattern family (open={result['pattern'][0]}, Asia={result['pattern'][1]}, London={result['pattern'][2]}) occurred {result['total_matches']} times. "
        f"Matched historical dates: {result['dates_text']}. "
        f"Out of those {result['total_matches']} matches, NY was up {result['up_count']} times, down {result['down_count']} times, "
        f"flat {result['flat_count']} times, and unknown {result['unknown_count']} times."
    )


def get_latest_day_snapshot(df: pd.DataFrame, flat_tolerance: float = 0.02, session_hours: Dict[str, Tuple[int, int]] | None = None) -> Tuple[str, str, str, pd.DataFrame]:
    if session_hours is None:
        session_hours = SESSION_WINDOWS

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    latest_date = frame["timestamp"].dt.date.max()
    day_frame = frame[frame["timestamp"].dt.date == latest_date].sort_values("timestamp").reset_index(drop=True)
    if day_frame.empty:
        raise ValueError(f"No data found for the latest date: {latest_date}")

    previous_day = frame[frame["timestamp"].dt.date < latest_date]
    prev_close = float(previous_day.iloc[-1]["close"]) if not previous_day.empty else None

    if prev_close is not None and prev_close != 0:
        open_label = classify_move((float(day_frame.iloc[0]["open"]) - prev_close) / prev_close, flat_tolerance)
    else:
        open_label = "unknown"

    session_snapshot = {}
    for session_name in ["asia", "london", "ny"]:
        move, label = _session_move_for_day(day_frame, session_name, session_hours)
        session_snapshot[session_name] = label

    asia_label = session_snapshot.get("asia", "unknown")
    london_label = session_snapshot.get("london", "unknown")
    latest_snapshot = pd.DataFrame(
        [{
            "date": latest_date,
            "open_label": open_label,
            "asia_label": asia_label,
            "london_label": london_label,
            "ny_label": session_snapshot.get("ny", "unknown"),
            "pattern": (open_label, asia_label, london_label),
        }]
    )
    return open_label, asia_label, london_label, latest_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Pattern-match the NY session outcome from historical session behavior.")
    parser.add_argument("--csv", default="binance_1h_history.csv", help="Path to the OHLCV CSV file.")
    parser.add_argument("--flat-tolerance", type=float, default=0.01, help="Move percentage threshold for flat/UP/DOWN classification.")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    history = build_session_history(df, flat_tolerance=args.flat_tolerance)

    if history.empty:
        raise ValueError(f"No usable session history was built from {csv_path}.")

    open_label, asia_label, london_label, today_snapshot = get_latest_day_snapshot(df, flat_tolerance=args.flat_tolerance)
    summary = build_pattern_summary(history, open_label, asia_label, london_label, flat_tolerance=args.flat_tolerance)

    print(f"Historical data rows used: {len(history)}")
    print(f"Today's pattern: open={open_label}, Asia={asia_label}, London={london_label}")
    print()
    print("Pattern match summary:")

    if summary["status"] == "no_data":
        print(f"- {summary['message']}")
    elif summary["status"] == "no_match":
        print(f"- {summary['message']}")
    else:
        print(f"- NY-session outlook: {summary['best_label']}")
        print(f"- Historical basis: {summary['historical_days']} days")
        print(f"- Pattern family: open={summary['pattern'][0]}, Asia={summary['pattern'][1]}, London={summary['pattern'][2]}")
        print(f"- Match count: {summary['total_matches']} times")
        print(f"- Matched dates: {summary['dates_text']}")
        print(f"- NY outcomes: up {summary['up_count']}, down {summary['down_count']}, flat {summary['flat_count']}, unknown {summary['unknown_count']}")


if __name__ == "__main__":
    main()

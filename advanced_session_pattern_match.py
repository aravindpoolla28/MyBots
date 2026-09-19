from __future__ import annotations

import argparse
import warnings
from typing import Dict, Iterable, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

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


def _session_move_for_day(day_frame: pd.DataFrame, session_name: str) -> Tuple[float, str]:
    """Get session move and classification for a given day."""
    start_hour, end_hour = SESSION_WINDOWS[session_name]
    session_rows = day_frame[
        (day_frame["timestamp"].dt.hour >= start_hour) & 
        (day_frame["timestamp"].dt.hour <= end_hour)
    ]
    
    if session_rows.empty:
        return 0.0, "unknown"
    
    session_open = float(session_rows.iloc[0]["open"])
    session_close = float(session_rows.iloc[-1]["close"])
    move = (session_close - session_open) / session_open if session_open else 0.0
    return move, classify_move(move)


def detect_liquidity_sweep(day_frame: pd.DataFrame, session_name: str) -> Dict[str, any]:
    """Detect liquidity sweeps using price action analysis."""
    start_hour, end_hour = SESSION_WINDOWS[session_name]
    session_rows = day_frame[
        (day_frame["timestamp"].dt.hour >= start_hour) & 
        (day_frame["timestamp"].dt.hour <= end_hour)
    ]
    
    if session_rows.empty:
        return {"sweep_detected": False, "sweep_type": None, "level": None, "strength": 0.0}
    
    # Get session OHLC
    session_high = float(session_rows["high"].max())
    session_low = float(session_rows["low"].min())
    session_open = float(session_rows.iloc[0]["open"])
    session_close = float(session_rows.iloc[-1]["close"])
    
    # Detect sweeps
    sweep_detected = False
    sweep_type = None
    level = None
    strength = 0.0
    
    # Check for bullish sweep (price tests low and closes above)
    if session_low < session_open and session_close > session_open:
        # Calculate sweep strength
        sweep_range = session_open - session_low
        total_range = session_high - session_low
        strength = sweep_range / total_range if total_range > 0 else 0.0
        
        if strength > 0.3:  # Significant sweep
            sweep_detected = True
            sweep_type = "bullish_sweep"
            level = session_low
    
    # Check for bearish sweep (price tests high and closes below)
    elif session_high > session_open and session_close < session_open:
        sweep_range = session_high - session_open
        total_range = session_high - session_low
        strength = sweep_range / total_range if total_range > 0 else 0.0
        
        if strength > 0.3:
            sweep_detected = True
            sweep_type = "bearish_sweep"
            level = session_high
    
    return {
        "sweep_detected": sweep_detected,
        "sweep_type": sweep_type,
        "level": level,
        "strength": strength,
        "session_high": session_high,
        "session_low": session_low,
        "session_open": session_open,
        "session_close": session_close
    }


def analyze_asia_london_interaction(day_frame: pd.DataFrame) -> Dict[str, any]:
    """Analyze Asia-London session interaction for ICT concepts."""
    asia_sweep = detect_liquidity_sweep(day_frame, "asia")
    london_sweep = detect_liquidity_sweep(day_frame, "london")
    
    # Get Asia session close price
    asia_start, asia_end = SESSION_WINDOWS["asia"]
    asia_rows = day_frame[
        (day_frame["timestamp"].dt.hour >= asia_start) & 
        (day_frame["timestamp"].dt.hour <= asia_end)
    ]
    
    asia_close = float(asia_rows.iloc[-1]["close"]) if not asia_rows.empty else 0.0
    
    # Get London session open price
    london_start, london_end = SESSION_WINDOWS["london"]
    london_rows = day_frame[
        (day_frame["timestamp"].dt.hour >= london_start) & 
        (day_frame["timestamp"].dt.hour <= london_end)
    ]
    
    london_open = float(london_rows.iloc[0]["open"]) if not london_rows.empty else 0.0
    
    # Analyze interaction
    interaction_type = None
    interaction_strength = 0.0
    
    if asia_sweep["sweep_detected"] and london_open > 0:
        # Price swept Asia low and opened in London
        price_move = london_open - asia_close
        if price_move > 0:  # Bullish sweep
            interaction_type = "bullish_liquidity_sweep"
            interaction_strength = min(price_move / asia_close, 0.5)  # Cap at 50%
        elif price_move < 0:  # Bearish sweep
            interaction_type = "bearish_liquidity_sweep"
            interaction_strength = min(abs(price_move) / asia_close, 0.5)
    
    return {
        "interaction_type": interaction_type,
        "interaction_strength": interaction_strength,
        "asia_sweep": asia_sweep,
        "london_sweep": london_sweep,
        "asia_close": asia_close,
        "london_open": london_open
    }


def extract_advanced_features(day_frame: pd.DataFrame) -> Dict[str, any]:
    """Extract advanced features for machine learning."""
    features = {}
    
    # Basic session metrics
    for session_name in ["asia", "london", "ny"]:
        start_hour, end_hour = SESSION_WINDOWS[session_name]
        session_rows = day_frame[
            (day_frame["timestamp"].dt.hour >= start_hour) & 
            (day_frame["timestamp"].dt.hour <= end_hour)
        ]
        
        if not session_rows.empty:
            session_open = float(session_rows.iloc[0]["open"])
            session_close = float(session_rows.iloc[-1]["close"])
            session_high = float(session_rows["high"].max())
            session_low = float(session_rows["low"].min())
            session_volume = float(session_rows["volume"].sum())
            
            features[f"{session_name}_move_pct"] = (session_close - session_open) / session_open if session_open else 0.0
            features[f"{session_name}_range_pct"] = (session_high - session_low) / session_open if session_open else 0.0
            features[f"{session_name}_close_position"] = (session_close - session_low) / (session_high - session_low) if (session_high - session_low) > 0 else 0.5
            features[f"{session_name}_volume"] = session_volume
            
            # Session volatility (standard deviation of hourly returns)
            hourly_returns = session_rows["close"].pct_change().dropna()
            features[f"{session_name}_volatility"] = hourly_returns.std() if len(hourly_returns) > 1 else 0.0
    
    # Cross-session interactions
    asia_sweep = detect_liquidity_sweep(day_frame, "asia")
    london_sweep = detect_liquidity_sweep(day_frame, "london")
    interaction = analyze_asia_london_interaction(day_frame)
    
    features.update({
        "asia_sweep_detected": 1 if asia_sweep["sweep_detected"] else 0,
        "london_sweep_detected": 1 if london_sweep["sweep_detected"] else 0,
        "interaction_strength": interaction["interaction_strength"],
        "interaction_type_bullish": 1 if interaction["interaction_type"] == "bullish_liquidity_sweep" else 0,
        "interaction_type_bearish": 1 if interaction["interaction_type"] == "bearish_liquidity_sweep" else 0,
    })
    
    return features


def build_advanced_session_history(df: pd.DataFrame, flat_tolerance: float = 0.02) -> pd.DataFrame:
    """Convert hourly OHLCV data into daily session classification rows with advanced features."""
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
            start_hour, end_hour = SESSION_WINDOWS[session_name]
            session_rows = day_frame[
                (day_frame["timestamp"].dt.hour >= start_hour) & 
                (day_frame["timestamp"].dt.hour <= end_hour)
            ]
            
            if session_rows.empty:
                session_snapshot[session_name] = {"move": 0.0, "label": "unknown"}
                continue
                
            session_open = float(session_rows.iloc[0]["open"])
            session_close = float(session_rows.iloc[-1]["close"])
            move = (session_close - session_open) / session_open if session_open else 0.0
            session_snapshot[session_name] = {"move": move, "label": classify_move(move)}

        # Extract advanced features
        advanced_features = extract_advanced_features(day_frame)
        
        # Get NY session outcome
        ny_label = session_snapshot.get("ny", {}).get("label", "unknown")
        
        record = {
            "date": day,
            "open_label": open_label,
            "asia_label": session_snapshot["asia"]["label"],
            "london_label": session_snapshot["london"]["label"],
            "ny_label": ny_label,
            "pattern": (open_label, session_snapshot["asia"]["label"], session_snapshot["london"]["label"]),
            **advanced_features
        }
        rows.append(record)

        prev_close = float(day_frame.iloc[-1]["close"])

    return pd.DataFrame(rows)


def train_ml_model(history: pd.DataFrame) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a machine learning model to predict NY session outcomes."""
    # Prepare features and target
    feature_columns = [
        "asia_move_pct", "london_move_pct", "ny_move_pct",
        "asia_range_pct", "london_range_pct", "ny_range_pct",
        "asia_close_position", "london_close_position", "ny_close_position",
        "asia_volatility", "london_volatility", "ny_volatility",
        "asia_sweep_detected", "london_sweep_detected",
        "interaction_strength", "interaction_type_bullish", "interaction_type_bearish"
    ]
    
    # Filter out rows with missing values
    history_clean = history.dropna(subset=feature_columns + ["ny_label"])
    
    if len(history_clean) < 10:
        raise ValueError("Not enough data to train ML model")
    
    X = history_clean[feature_columns]
    y = history_clean["ny_label"].map({"up": 1, "down": 0, "flat": 2})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ML Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=["down", "up", "flat"]))
    
    return model, scaler


def analyze_statistical_patterns(history: pd.DataFrame) -> Dict[str, any]:
    """Analyze statistical patterns and correlations."""
    patterns = {}
    
    # Analyze pattern frequencies
    pattern_groups = history.groupby("pattern")
    for pattern, group in pattern_groups:
        ny_outcomes = group["ny_label"].value_counts()
        patterns[str(pattern)] = {
            "total_days": len(group),
            "ny_up": ny_outcomes.get("up", 0),
            "ny_down": ny_outcomes.get("down", 0),
            "ny_flat": ny_outcomes.get("flat", 0),
            "ny_unknown": ny_outcomes.get("unknown", 0),
            "success_rate_up": ny_outcomes.get("up", 0) / len(group) if len(group) > 0 else 0,
            "success_rate_down": ny_outcomes.get("down", 0) / len(group) if len(group) > 0 else 0,
        }
    
    # Analyze sweep patterns
    sweep_patterns = history[history["asia_sweep_detected"] == 1]
    if not sweep_patterns.empty:
        ny_outcomes_sweep = sweep_patterns["ny_label"].value_counts()
        patterns["sweep_patterns"] = {
            "total_days": len(sweep_patterns),
            "ny_up": ny_outcomes_sweep.get("up", 0),
            "ny_down": ny_outcomes_sweep.get("down", 0),
            "ny_flat": ny_outcomes_sweep.get("flat", 0),
        }
    
    # Analyze interaction patterns
    interaction_patterns = history[history["interaction_type_bullish"] == 1]
    if not interaction_patterns.empty:
        ny_outcomes_interaction = interaction_patterns["ny_label"].value_counts()
        patterns["bullish_interaction"] = {
            "total_days": len(interaction_patterns),
            "ny_up": ny_outcomes_interaction.get("up", 0),
            "ny_down": ny_outcomes_interaction.get("down", 0),
            "ny_flat": ny_outcomes_interaction.get("flat", 0),
        }
    
    return patterns


def build_pattern_summary(history: pd.DataFrame, open_label: str, asia_label: str, london_label: str, flat_tolerance: float = 0.02) -> Dict[str, any]:
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
        # Use broader bucket matching
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

    # Add ML prediction if available
    ml_prediction = None
    if "ml_model" in globals() and "ml_scaler" in globals():
        try:
            # Prepare features for prediction
            features = extract_advanced_features(matches.iloc[0])
            feature_columns = [
                "asia_move_pct", "london_move_pct", "ny_move_pct",
                "asia_range_pct", "london_range_pct", "ny_range_pct",
                "asia_close_position", "london_close_position", "ny_close_position",
                "asia_volatility", "london_volatility", "ny_volatility",
                "asia_sweep_detected", "london_sweep_detected",
                "interaction_strength", "interaction_type_bullish", "interaction_type_bearish"
            ]
            X_pred = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
            X_pred_scaled = ml_scaler.transform(X_pred)
            ml_prediction = ml_model.predict(X_pred_scaled)[0]
            ml_prediction = "up" if ml_prediction == 1 else "down" if ml_prediction == 0 else "flat"
        except Exception as e:
            ml_prediction = f"Error: {str(e)}"

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
        "ml_prediction": ml_prediction,
    }


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


def summarize_pattern(history: pd.DataFrame, open_label: str, asia_label: str, london_label: str, flat_tolerance: float = 0.02) -> str:
    """Return a probabilistic summary for a pattern match on NY session behavior."""
    result = build_pattern_summary(history, open_label, asia_label, london_label, flat_tolerance)
    if result["status"] == "no_data":
        return result["message"]
    if result["status"] == "no_match":
        return result["message"]

    ml_info = f" ML Prediction: {result['ml_prediction']}" if result.get("ml_prediction") else ""
    
    return (
        f"Today's NY-session price action is most likely {result['best_label']} based on {result['historical_days']} historical days. "
        f"This {result['match_mode']} pattern family (open={result['pattern'][0]}, Asia={result['pattern'][1]}, London={result['pattern'][2]}) occurred {result['total_matches']} times. "
        f"Matched historical dates: {result['dates_text']}. "
        f"Out of those {result['total_matches']} matches, NY was up {result['up_count']} times, down {result['down_count']} times, "
        f"flat {result['flat_count']} times, and unknown {result['unknown_count']} times.{ml_info}"
    )


def get_latest_day_snapshot(df: pd.DataFrame, flat_tolerance: float = 0.02) -> Tuple[str, str, str, pd.DataFrame]:
    """Get the latest day's session snapshot with advanced features."""
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
        move, label = _session_move_for_day(day_frame, session_name)
        session_snapshot[session_name] = label

    asia_label = session_snapshot.get("asia", "unknown")
    london_label = session_snapshot.get("london", "unknown")
    
    # Get advanced features for latest day
    advanced_features = extract_advanced_features(day_frame)

    latest_snapshot = pd.DataFrame(
        [{
            "date": latest_date,
            "open_label": open_label,
            "asia_label": asia_label,
            "london_label": london_label,
            "ny_label": session_snapshot.get("ny", "unknown"),
            "pattern": (open_label, asia_label, london_label),
            **advanced_features
        }]
    )
    return open_label, asia_label, london_label, latest_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced Pattern-match the NY session outcome from historical session behavior with ICT concepts and ML models.")
    parser.add_argument("--csv", default="binance_1h_history.csv", help="Path to the OHLCV CSV file.")
    parser.add_argument("--flat-tolerance", type=float, default=0.01, help="Move percentage threshold for flat/UP/DOWN classification.")
    parser.add_argument("--train-ml", action="store_true", help="Train and use ML model for predictions.")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    history = build_advanced_session_history(df, flat_tolerance=args.flat_tolerance)

    if history.empty:
        raise ValueError(f"No usable session history was built from {csv_path}.")

    # Train ML model if requested
    global ml_model, ml_scaler
    ml_model = None
    ml_scaler = None
    
    if args.train_ml:
        try:
            ml_model, ml_scaler = train_ml_model(history)
            print("ML model trained successfully!")
        except Exception as e:
            print(f"Warning: Could not train ML model: {e}")

    open_label, asia_label, london_label, today_snapshot = get_latest_day_snapshot(df, flat_tolerance=args.flat_tolerance)
    summary = build_pattern_summary(history, open_label, asia_label, london_label, flat_tolerance=args.flat_tolerance)

    print(f"Historical data rows used: {len(history)}")
    print(f"Today's pattern: open={open_label}, Asia={asia_label}, London={london_label}")
    
    # Show advanced features for today
    print("\nToday's Advanced Features:")
    for col in today_snapshot.columns:
        if col not in ["date", "open_label", "asia_label", "london_label", "ny_label", "pattern"]:
            print(f"  {col}: {today_snapshot.iloc[0][col]:.4f}")
    
    print("\nPattern match summary:")

    if summary["status"] == "no_data":
        print(f"- {summary['message']}")
    elif summary["status"] == "no_match":
        print(f"- {summary['message']}")
    else:
        print(f"- NY-session outlook: {summary['best_label']}")
        if summary.get("ml_prediction"):
            print(f"- ML Model Prediction: {summary['ml_prediction']}")
        print(f"- Historical basis: {summary['historical_days']} days")
        print(f"- Pattern family: open={summary['pattern'][0]}, Asia={summary['pattern'][1]}, London={summary['pattern'][2]}")
        print(f"- Match count: {summary['total_matches']} times")
        print(f"- Matched dates: {summary['dates_text']}")
        print(f"- NY outcomes: up {summary['up_count']}, down {summary['down_count']}, flat {summary['flat_count']}, unknown {summary['unknown_count']}")

    # Show statistical patterns
    print("\nStatistical Pattern Analysis:")
    patterns = analyze_statistical_patterns(history)
    for pattern_name, pattern_data in patterns.items():
        print(f"\n{pattern_name.replace('_', ' ').title()}:")
        print(f"  Total days: {pattern_data.get('total_days', 'N/A')}")
        if 'success_rate_up' in pattern_data:
            print(f"  Success rate (up): {pattern_data['success_rate_up']:.2%}")
        if 'success_rate_down' in pattern_data:
            print(f"  Success rate (down): {pattern_data['success_rate_down']:.2%}")
        print(f"  NY outcomes: up {pattern_data.get('ny_up', 0)}, down {pattern_data.get('ny_down', 0)}, flat {pattern_data.get('ny_flat', 0)}")


if __name__ == "__main__":
    main()
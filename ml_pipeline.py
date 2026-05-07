import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


STATUS_FEATURE_COLUMNS = [
    "current_stock",
    "daily_demand",
    "expiry_days_left",
    "season",
    "category",
    "demand_trend",
    "lead_time_days",
]
STATUS_TARGET_COLUMN = "inventory_status"
# These fields are normalized/encoded before training and inference.
CATEGORICAL_COLUMNS = ["season", "category", "demand_trend"]
LAG_COLUMNS = ["lag_1", "lag_2", "lag_3"]
GROUP_COLUMNS = ["store_name", "medicine_name"]
# Keep forecast output shape stable even when no rows are produced.
FORECAST_OUTPUT_COLUMNS = ["store_name", "medicine_name", "forecast_day", "date", "predicted_demand"]


SEASON_SYNONYMS = {
    "winter": "winter",
    "summer": "summer",
    "monsoon": "monsoon",
    "rainy": "monsoon",
}

CATEGORY_SYNONYMS = {
    "cold": "respiratory",
    "respiratory": "respiratory",
    "cardio": "cardiovascular",
    "cardiovascular": "cardiovascular",
    "antibiotic": "antibiotic",
    "painkiller": "pain relief",
    "pain relief": "pain relief",
    "diabetes": "diabetes",
    "thyroid": "thyroid",
    "gastro": "gastro",
    "allergy": "allergy",
    "neurological": "neurological",
    "emergency": "emergency",
    "immunology": "immunology",
    "psychiatric": "psychiatric",
}

DEMAND_TREND_SYNONYMS = {
    "decreasing": "decreasing",
    "stable": "stable",
    "increasing": "increasing",
}


def _required_columns(df: pd.DataFrame, columns: List[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize category strings so training and runtime inputs stay consistent.
    out = df.copy()
    if "season" in out.columns:
        out["season"] = out["season"].astype(str).str.strip().str.lower().map(
            lambda value: SEASON_SYNONYMS.get(value, value)
        )
    if "category" in out.columns:
        out["category"] = out["category"].astype(str).str.strip().str.lower().map(
            lambda value: CATEGORY_SYNONYMS.get(value, value)
        )
    if "demand_trend" in out.columns:
        out["demand_trend"] = out["demand_trend"].astype(str).str.strip().str.lower().map(
            lambda value: DEMAND_TREND_SYNONYMS.get(value, value)
        )
    return out


def build_category_maps(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    # Build deterministic maps from category text to integer ids.
    maps: Dict[str, Dict[str, int]] = {}
    for col in CATEGORICAL_COLUMNS:
        unique_values = sorted(df[col].dropna().unique().tolist())
        maps[col] = {value: idx for idx, value in enumerate(unique_values)}
    return maps


def encode_categoricals(
    df: pd.DataFrame, category_maps: Dict[str, Dict[str, int]], unknown_value: int = -1
) -> pd.DataFrame:
    # Unknown labels are sent to -1 so inference does not crash.
    encoded = df.copy()
    for col in CATEGORICAL_COLUMNS:
        encoded[col] = encoded[col].map(category_maps[col]).fillna(unknown_value).astype(int)
    return encoded


def validate_input_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    # Shared schema contract for both training and inference APIs.
    required = [
        "store_name",
        "medicine_name",
        "date",
        "current_stock",
        "daily_demand",
        "expiry_days_left",
        "season",
        "category",
        "demand_trend",
        "lead_time_days",
        "unit_price",
        "reorder_fee",
        "distance_from_distributor_miles",
    ]
    if require_target:
        required.append(STATUS_TARGET_COLUMN)

    _required_columns(df, required, "input schema")


def train_status_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[RandomForestClassifier, Dict]:
    # Model 1: classification model that predicts inventory status.
    validate_input_schema(df, require_target=True)
    normalized = normalize_categoricals(df)

    category_maps = build_category_maps(normalized)
    encoded = encode_categoricals(normalized, category_maps)
    X = encoded[STATUS_FEATURE_COLUMNS]
    y = encoded[STATUS_TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # RandomForest is robust for mixed operational features and small-to-medium tabular data.
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, pred))

    metadata = {
        "feature_columns": STATUS_FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "category_maps": category_maps,
        "metrics": {"accuracy": accuracy},
    }
    return model, metadata


def predict_status(df: pd.DataFrame, model: RandomForestClassifier, metadata: Dict) -> pd.Series:
    # Runtime inference path used by the website analyze endpoint.
    _required_columns(df, STATUS_FEATURE_COLUMNS, "status prediction")
    normalized = normalize_categoricals(df)
    encoded = encode_categoricals(normalized, metadata["category_maps"])
    predictions = model.predict(encoded[metadata["feature_columns"]])
    return pd.Series(predictions, index=df.index, name="predicted_status")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Build lag_1/lag_2/lag_3 per store+medicine time series.
    _required_columns(df, GROUP_COLUMNS + ["date", "daily_demand"], "lag feature generation")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(GROUP_COLUMNS + ["date"])

    for offset, col_name in enumerate(LAG_COLUMNS, start=1):
        out[col_name] = out.groupby(GROUP_COLUMNS)["daily_demand"].shift(offset)
    return out


def train_demand_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[RandomForestRegressor, Dict]:
    # Model 2: demand forecasting model based on recent lagged demand.
    validate_input_schema(df, require_target=False)
    lagged = add_lag_features(df)
    forecast_df = lagged.dropna(subset=LAG_COLUMNS).copy()
    if forecast_df.empty:
        raise ValueError("Not enough demand history to train demand model.")

    X = forecast_df[LAG_COLUMNS]
    y = forecast_df["daily_demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    metadata = {"lag_columns": LAG_COLUMNS, "metrics": {"mae": mae}}
    return model, metadata


def forecast_demand_all_groups(
    df: pd.DataFrame, model: RandomForestRegressor, horizon_days: int = 3
) -> pd.DataFrame:
    # Multi-store forecast used directly by frontend dashboards.
    _required_columns(df, GROUP_COLUMNS + ["date", "daily_demand"], "group forecasting")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(GROUP_COLUMNS + ["date"])

    results = []
    for (store_name, medicine_name), group in out.groupby(GROUP_COLUMNS):
        history = group["daily_demand"].tail(3).tolist()
        if len(history) < 3:
            # Skip groups without enough history to build lag input.
            continue

        lag_1, lag_2, lag_3 = history[-1], history[-2], history[-3]
        base_date = group["date"].iloc[-1]

        for day in range(1, horizon_days + 1):
            next_input = pd.DataFrame([[lag_1, lag_2, lag_3]], columns=LAG_COLUMNS)
            predicted = float(model.predict(next_input)[0])
            predicted = max(predicted, 1.0)
            results.append(
                {
                    "store_name": store_name,
                    "medicine_name": medicine_name,
                    "forecast_day": day,
                    "date": (base_date + dt.timedelta(days=day)).date().isoformat(),
                    "predicted_demand": round(predicted, 2),
                }
            )
            lag_3, lag_2, lag_1 = lag_2, lag_1, predicted

    return pd.DataFrame(results, columns=FORECAST_OUTPUT_COLUMNS)

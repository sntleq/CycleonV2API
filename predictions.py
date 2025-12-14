import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
import httpx
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

router = APIRouter(prefix="/predictions")

CACHE_DIR = Path("cache/predictions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CACHE_DIR = Path("cache/prices")
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_URL = "https://api.weirdgloop.org/exchange/history/rs/all?id={item_id}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RETRIES = 5
RETRY_DELAY = 1


# ---------- DATA FETCH ----------
async def fetch_price_graph(client: httpx.AsyncClient, item_id: str):
    url = GRAPH_URL.format(item_id=item_id)
    last_error = None

    for attempt in range(RETRIES):
        try:
            r = await client.get(url, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            price_history = data.get(item_id)
            if not price_history:
                raise RuntimeError("No price history")
            return price_history
        except Exception as e:
            last_error = e
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"Failed fetching prices: {last_error}")


# ---------- DATA PREP ----------
def prepare_dataframe(item_id: str, price_data: list) -> pd.DataFrame:
    records = []
    for entry in price_data:
        if entry.get("price") and entry.get("timestamp"):
            records.append({
                "ds": pd.to_datetime(entry["timestamp"], unit="ms"),
                "y": entry["price"]
            })

    df = pd.DataFrame(records).sort_values("ds").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lags
    lags = [1, 2, 3, 7, 14, 30, 90, 180, 365]
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics
    windows = [7, 14, 30, 90, 180]
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].rolling(w).std()

    # Expanding statistics for long-term trend
    df["expanding_mean"] = df["y"].expanding().mean()
    df["expanding_std"] = df["y"].expanding().std()

    # Calendar features
    df["dow"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)

    # Trend feature
    df["time_idx"] = np.arange(len(df))

    return df.dropna().reset_index(drop=True)


# ---------- MODEL ----------
def train_and_predict(df: pd.DataFrame, horizon: int = 90, log_transform: bool = True) -> pd.DataFrame:
    df_feat = add_features(df)
    features = [c for c in df_feat.columns if c not in ("ds", "y")]

    # Optional log-transform for stabilizing exponential growth
    y = np.log1p(df_feat["y"]) if log_transform else df_feat["y"]

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    model.fit(df_feat[features], y)

    forecast = []
    last_row = df_feat.iloc[-1:].copy()
    prev_values = df["y"].tolist()

    for _ in range(horizon):
        x_pred = last_row[features]
        y_pred = model.predict(x_pred)[0]
        y_pred = np.expm1(y_pred) if log_transform else y_pred
        y_pred = max(0, y_pred)  # no negative prices
        forecast.append(y_pred)
        prev_values.append(y_pred)

        # Update lags
        for lag in [1, 2, 3, 7, 14, 30, 90, 180, 365]:
            last_row[f"lag_{lag}"] = prev_values[-lag] if len(prev_values) >= lag else prev_values[-1]

        # Update rolling stats
        for w in [7, 14, 30, 90, 180]:
            window = prev_values[-w:] if len(prev_values) >= w else prev_values
            last_row[f"roll_mean_{w}"] = np.mean(window)
            last_row[f"roll_std_{w}"] = np.std(window)

        # Update expanding stats
        last_row["expanding_mean"] = np.mean(prev_values)
        last_row["expanding_std"] = np.std(prev_values)

        # Increment trend index
        last_row["time_idx"] += 1

    last_date = df["ds"].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast))
    forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": forecast})

    df["yhat"] = df["y"]
    return pd.concat([df[["ds", "yhat"]], forecast_df], ignore_index=True)


# ---------- OUTPUT FORMAT ----------
def format_predictions(forecast: pd.DataFrame, df_actual: pd.DataFrame, item_id: str, period: int):
    today = datetime.now().date()
    start = today - timedelta(days=max(period, 7))
    end = today + timedelta(days=period)

    df_actual["date"] = df_actual["ds"].dt.date
    actual_map = dict(zip(df_actual["date"], df_actual["y"]))

    forecast["date"] = forecast["ds"].dt.date
    forecast = forecast[(forecast["date"] >= start) & (forecast["date"] <= end)]

    result = []
    for _, row in forecast.iterrows():
        d = row["date"]
        price = actual_map.get(d, int(round(row["yhat"])))
        result.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            item_id: int(round(price))
        })

    return result


# ---------- API ----------
@router.get("/{item_id}")
async def get_price_prediction(
        item_id: str,
        period: int = Query(default=30, ge=1, le=365)
):
    try:
        cache_file = CACHE_DIR / f"item_{item_id}_prediction_full.json"
        api_cache_file = PRICES_CACHE_DIR / f"item_{item_id}_api_data.json"

        if cache_file.exists() and api_cache_file.exists():
            forecast = pd.DataFrame(json.loads(cache_file.read_text()))
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            df_actual = prepare_dataframe(item_id, json.loads(api_cache_file.read_text()))
            return format_predictions(forecast, df_actual, item_id, period)

        async with httpx.AsyncClient() as client:
            price_data = await fetch_price_graph(client, item_id)

        api_cache_file.write_text(json.dumps(price_data, indent=2))
        df = prepare_dataframe(item_id, price_data)
        forecast = train_and_predict(df, horizon=90, log_transform=True)

        forecast_cache = forecast.copy()
        forecast_cache["ds"] = forecast_cache["ds"].dt.strftime("%Y-%m-%d")
        cache_file.write_text(json.dumps(forecast_cache.to_dict("records"), indent=2))

        return format_predictions(forecast, df, item_id, period)

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

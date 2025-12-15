import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
import httpx
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

router = APIRouter(prefix="/predictions")

# ----------------- CACHE -----------------
CACHE_DIR = Path("cache/predictions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CACHE_DIR = Path("cache/prices")
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- API -----------------
GRAPH_URL = "https://api.weirdgloop.org/exchange/history/rs/all?id={item_id}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RETRIES = 5
RETRY_DELAY = 1


# ----------------- FETCH -----------------
async def fetch_price_graph(client: httpx.AsyncClient, item_id: str):
    url = GRAPH_URL.format(item_id=item_id)
    last_error = None

    for attempt in range(RETRIES):
        try:
            r = await client.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            history = data.get(item_id)
            if not history:
                raise RuntimeError("No price history")
            return history
        except Exception as e:
            last_error = e
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"Failed fetching prices: {last_error}")


# ----------------- DATA PREP -----------------
def prepare_dataframe(price_data: list) -> pd.DataFrame:
    records = []
    for e in price_data:
        if e.get("price") and e.get("timestamp"):
            records.append({
                "ds": pd.to_datetime(e["timestamp"], unit="ms"),
                "y": float(e["price"])
            })

    df = pd.DataFrame(records)
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lags (cheap but powerful)
    for lag in (1, 7, 30, 90):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling means (trend detection)
    for w in (7, 30, 90):
        df[f"roll_mean_{w}"] = df["y"].rolling(w).mean()

    # Explicit time trend
    df["time_idx"] = np.arange(len(df))

    return df.dropna().reset_index(drop=True)


# ----------------- MODEL -----------------
def train_and_predict(df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    df_feat = add_features(df)
    features = [c for c in df_feat.columns if c not in ("ds", "y")]

    # Stabilize growth
    y = np.log1p(df_feat["y"])

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=1,               # 🚨 critical for Railway
        random_state=42
    )

    model.fit(df_feat[features], y)

    forecast = []
    last = df_feat.iloc[-1:].copy()
    history = df_feat["y"].tolist()

    for _ in range(horizon):
        y_pred = model.predict(last[features])[0]
        y_pred = max(0.0, np.expm1(y_pred))
        forecast.append(y_pred)
        history.append(y_pred)

        # update features
        for lag in (1, 7, 30, 90):
            last[f"lag_{lag}"] = history[-lag]

        for w in (7, 30, 90):
            last[f"roll_mean_{w}"] = np.mean(history[-w:])

        last["time_idx"] += 1

    future_dates = pd.date_range(
        start=df["ds"].max() + timedelta(days=1),
        periods=horizon
    )

    hist = df[["ds", "y"]].rename(columns={"y": "yhat"})
    fut = pd.DataFrame({"ds": future_dates, "yhat": forecast})

    return pd.concat([hist, fut], ignore_index=True)


# ----------------- FORMAT -----------------
def format_predictions(
    forecast: pd.DataFrame,
    df_actual: pd.DataFrame,
    item_id: str,
    period: int
):
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=max(period, 7))
    end = today + timedelta(days=period)

    df_actual["date"] = df_actual["ds"].dt.date
    actual_map = dict(zip(df_actual["date"], df_actual["y"]))

    forecast["date"] = forecast["ds"].dt.date
    forecast = forecast[
        (forecast["date"] >= start) &
        (forecast["date"] <= end)
    ]

    out = []
    for _, r in forecast.iterrows():
        price = actual_map.get(r["date"], r["yhat"])
        out.append({
            "date": r["ds"].strftime("%Y-%m-%d"),
            item_id: int(round(price))
        })

    return out


# ----------------- ENDPOINT -----------------
@router.get("/{item_id}")
async def get_price_prediction(
    item_id: str,
    period: int = Query(default=30, ge=1, le=90)
):
    try:
        cache_file = CACHE_DIR / f"{item_id}_forecast.json"
        api_cache_file = PRICES_CACHE_DIR / f"{item_id}_prices.json"

        # ---- CACHE HIT ----
        if cache_file.exists() and api_cache_file.exists():
            forecast = pd.DataFrame(json.loads(cache_file.read_text()))
            forecast["ds"] = pd.to_datetime(forecast["ds"])

            df_actual = prepare_dataframe(
                json.loads(api_cache_file.read_text())
            )

            return format_predictions(
                forecast, df_actual, item_id, period
            )

        # ---- FETCH DATA ----
        async with httpx.AsyncClient() as client:
            price_data = await fetch_price_graph(client, item_id)

        api_cache_file.write_text(json.dumps(price_data))

        df = prepare_dataframe(price_data)

        # 🚨 offload CPU training
        forecast = await asyncio.to_thread(
            train_and_predict, df, 90
        )

        cache_df = forecast.copy()
        cache_df["ds"] = cache_df["ds"].dt.strftime("%Y-%m-%d")
        cache_file.write_text(
            json.dumps(cache_df.to_dict("records"))
        )

        return format_predictions(
            forecast, df, item_id, period
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

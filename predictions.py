import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
import httpx
import pandas as pd
import timesfm

router = APIRouter(prefix="/predictions")

CACHE_DIR = Path("cache/predictions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CACHE_DIR = Path("cache/prices")
PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_URL = "https://api.weirdgloop.org/exchange/history/rs/all?id={item_id}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RETRIES = 5
RETRY_DELAY = 1  # seconds

# Global variable to hold the model (lazy loaded)
_tfm_model = None


def get_model():
    """Lazy load the TimesFM model."""
    global _tfm_model
    if _tfm_model is None:
        print("Loading TimesFM model...")
        _tfm_model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=8,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )
        print("TimesFM model loaded successfully!")
    return _tfm_model


async def fetch_price_graph(client: httpx.AsyncClient, item_id: str):
    """Fetch full price history from Weirdgloop API."""
    url = GRAPH_URL.format(item_id=item_id)
    last_error = None

    for attempt in range(RETRIES):
        try:
            r = await client.get(url, headers=HEADERS)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            if not r.content:
                raise RuntimeError("Empty response")

            data = r.json()
            # Response format: {"item_id": [{"id": "...", "price": ..., "timestamp": ...}, ...]}
            price_history = data.get(item_id)
            if not price_history:
                raise RuntimeError(f"No price history for item {item_id}")

            return price_history

        except Exception as e:
            last_error = e
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"Failed after {RETRIES} retries for {item_id}: {last_error}")


def prepare_dataframe(item_id: str, price_data: list) -> pd.DataFrame:
    """Prepare DataFrame from Weirdgloop API data."""
    # Convert API data to format for TimesFM
    # Format: [{"id": "4151", "price": 1500000, "volume": null, "timestamp": 1211328000000}, ...]
    api_records = []
    for entry in price_data:
        price = entry.get("price")
        timestamp = entry.get("timestamp")

        if timestamp and price:
            # Timestamp is in milliseconds
            date = pd.to_datetime(timestamp, unit='ms')
            api_records.append({"ds": date, "y": price})

    df = pd.DataFrame(api_records)
    df = df.sort_values("ds").reset_index(drop=True)

    # Limit to last 365 days to reduce memory usage
    if len(df) > 365:
        df = df.tail(365).reset_index(drop=True)

    # Data quality check
    print(f"Item {item_id} price stats (Weirdgloop API):")
    print(f"  Min: {df['y'].min():,.0f}")
    print(f"  Max: {df['y'].max():,.0f}")
    print(f"  Mean: {df['y'].mean():,.0f}")
    print(f"  Count: {len(df)}")
    print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")

    return df


def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Use TimesFM to generate predictions."""
    # Get the model (lazy loaded)
    tfm = get_model()

    # Prepare input for TimesFM (just the price values)
    price_series = df['y'].values

    # TimesFM expects frequency category: 0=high freq (daily), 1=medium (weekly/monthly), 2=low (quarterly/yearly)
    # Daily data = frequency 0
    point_forecast, quantile_forecast = tfm.forecast(
        [price_series],  # List of time series (we have 1)
        freq=[0],  # Daily frequency
    )

    # Debug: Check what we got back
    print(f"Forecast shape: {point_forecast[0].shape}")

    # Get the actual forecast length
    forecast_length = len(point_forecast[0])

    # Create forecast dataframe
    last_date = df['ds'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_length, freq='D')

    # Combine historical and forecast data (only yhat, no upper/lower bounds)
    historical_df = df.copy()
    historical_df['yhat'] = df['y']  # For historical, yhat = actual

    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': point_forecast[0],
    })

    # Combine historical and future
    full_forecast = pd.concat([
        historical_df[['ds', 'yhat']],
        forecast_df
    ], ignore_index=True)

    return full_forecast


def format_predictions(forecast: pd.DataFrame, df_actual: pd.DataFrame, item_id: str, period: int) -> list:
    """Format predictions: show actual values for history, predictions for future."""
    today = datetime.now().date()

    # Calculate date range: max(period, 7) days before today to period days after today
    days_before = max(period, 7)
    start_date = today - timedelta(days=days_before)
    end_date = today + timedelta(days=period)

    # Create a dict of actual values for quick lookup
    df_actual["date"] = pd.to_datetime(df_actual["ds"]).dt.date
    actual_values = dict(zip(df_actual["date"], df_actual["y"]))

    # Filter forecast to requested date range
    forecast["date"] = pd.to_datetime(forecast["ds"]).dt.date
    mask = (forecast["date"] >= start_date) & (forecast["date"] <= end_date)
    filtered = forecast[mask].copy()

    # Format output: use actual values for historical dates, predictions for future
    result = []
    for _, row in filtered.iterrows():
        row_date = row["date"]
        # Use actual API value if we have it, otherwise use TimesFM prediction
        if row_date in actual_values:
            price = int(round(actual_values[row_date]))
        else:
            price = int(round(row["yhat"]))

        result.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            item_id: price
        })

    return result


@router.get("/{item_id}")
async def get_price_prediction(
        item_id: str,
        period: int = Query(default=30, ge=1, le=365,
                            description="Number of days to show in output")
):
    """
    Get price predictions for an item using TimesFM.

    - **item_id**: The RuneScape item ID
    - **period**: Number of days to show in output (default: 30, max: 365)

    Returns predictions from max(period, 7) days before today to period days after today.
    Historical dates show actual API values, future dates show TimesFM predictions.
    """
    try:
        # Cache files
        cache_file = CACHE_DIR / f"item_{item_id}_prediction_full.json"
        api_cache_file = PRICES_CACHE_DIR / f"item_{item_id}_api_data.json"

        # Check if both caches exist
        if cache_file.exists() and api_cache_file.exists():
            # Load cached forecast
            with cache_file.open("r", encoding="utf-8") as f:
                cached_forecast = json.load(f)
            forecast_df = pd.DataFrame(cached_forecast)
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

            # Load cached API data
            with api_cache_file.open("r", encoding="utf-8") as f:
                price_data = json.load(f)
            df_actual = prepare_dataframe(item_id, price_data)

            # Format based on requested period
            predictions = format_predictions(forecast_df, df_actual, item_id, period)
            return predictions

        # Fetch fresh data from API
        async with httpx.AsyncClient() as client:
            price_data = await fetch_price_graph(client, item_id)

        # Cache the API data
        with api_cache_file.open("w", encoding="utf-8") as f:
            json.dump(price_data, f, ensure_ascii=False, indent=2)

        # Prepare DataFrame
        df = prepare_dataframe(item_id, price_data)

        # Train model and predict
        forecast = train_and_predict(df)

        # Cache the full forecast with all columns
        forecast_cache = forecast.copy()
        forecast_cache["ds"] = forecast_cache["ds"].dt.strftime("%Y-%m-%d")
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(forecast_cache.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        # Format predictions based on requested period
        predictions = format_predictions(forecast, df, item_id, period)

        return predictions

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Prediction failed: {str(e)}"
        )
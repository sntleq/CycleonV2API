import asyncio
import json
from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter(prefix="/rankings")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

ITEM_RANKINGS_CACHE = CACHE_DIR / "item_rankings.json"
WEIRDGLOOP_URL = "https://api.weirdgloop.org/exchange/history/rs/latest"
HEADERS = {"User-Agent": "Mozilla/5.0"}

CONCURRENCY_LIMIT = 4
RETRIES = 5
RETRY_DELAY = 1  # seconds
BATCH_SIZE = 100
TOP_N_CACHE = 20

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# Load item rankings cache at startup
if ITEM_RANKINGS_CACHE.exists():
    with ITEM_RANKINGS_CACHE.open("r", encoding="utf-8") as f:
        ITEM_RANKINGS = json.load(f)
else:
    ITEM_RANKINGS = {
        "by_price": [],
        "by_volume": [],
        "details": {"by_price": [], "by_volume": []},
        "timestamp": None
    }


async def fetch_price_data_batch(client: httpx.AsyncClient, item_ids: List[str]) -> Dict:
    """Fetch price data for a batch of items from Weirdgloop API."""
    ids_param = "|".join(item_ids)
    url = f"{WEIRDGLOOP_URL}?id={ids_param}"
    last_error = None

    async with semaphore:
        for attempt in range(RETRIES):
            try:
                r = await client.get(url, headers=HEADERS, timeout=30.0)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                if not r.content:
                    raise RuntimeError("Empty response")

                return r.json()

            except Exception as e:
                last_error = e
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"Failed after {RETRIES} retries: {last_error}")


async def fetch_all_price_data(item_ids: List[str]):
    """Fetch price data for all items and cache top N by price and volume."""
    # Split into batches of 100
    batches = [item_ids[i:i + BATCH_SIZE] for i in range(0, len(item_ids), BATCH_SIZE)]

    # Fetch all batches in parallel
    async with httpx.AsyncClient() as client:
        tasks = [fetch_price_data_batch(client, batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine all results
    all_price_data = {}
    for result in results:
        if isinstance(result, Exception):
            continue  # Skip failed batches
        if isinstance(result, dict):
            all_price_data.update(result)

    # Convert to list and sort by price and volume
    items_list = []
    for item_id, data in all_price_data.items():
        if data and isinstance(data, dict):
            price = data.get("price", 0)
            volume = data.get("volume", 0)
            if price and volume:  # Only include items with valid data
                items_list.append({
                    "id": item_id,
                    "price": price,
                    "volume": volume
                })

    # Sort and cache top N
    top_by_price = sorted(items_list, key=lambda x: x["price"], reverse=True)[:TOP_N_CACHE]
    top_by_volume = sorted(items_list, key=lambda x: x["volume"], reverse=True)[:TOP_N_CACHE]

    # Cache results
    global ITEM_RANKINGS
    ITEM_RANKINGS = {
        "by_price": [item["id"] for item in top_by_price],
        "by_volume": [item["id"] for item in top_by_volume],
        "details": {
            "by_price": top_by_price,
            "by_volume": top_by_volume
        },
        "timestamp": asyncio.get_event_loop().time()
    }

    with ITEM_RANKINGS_CACHE.open("w", encoding="utf-8") as f:
        json.dump(ITEM_RANKINGS, f, ensure_ascii=False, indent=2)

    return ITEM_RANKINGS


@router.get("/price")
async def get_top_by_price(n: int = 10):
    """Get top N items by highest price."""
    if not ITEM_RANKINGS.get("by_price"):
        # Import here to avoid circular dependency
        from items import get_all_items
        all_items = await get_all_items()
        item_ids = [item["value"] for item in all_items]
        await fetch_all_price_data(item_ids)

    top_n_details = ITEM_RANKINGS["details"]["by_price"][:n]
    result = [{"id": item["id"], "price": item["price"]} for item in top_n_details]
    return result


@router.get("/volume")
async def get_top_by_volume(n: int = 10):
    """Get top N items by highest volume."""
    if not ITEM_RANKINGS.get("by_volume"):
        # Import here to avoid circular dependency
        from items import get_all_items
        all_items = await get_all_items()
        item_ids = [item["value"] for item in all_items]
        await fetch_all_price_data(item_ids)

    top_n_details = ITEM_RANKINGS["details"]["by_volume"][:n]
    result = [{"id": item["id"], "volume": item["volume"]} for item in top_n_details]
    return result


@router.post("/refresh")
async def refresh_rankings():
    """Force refresh of price rankings cache."""
    try:
        # Import here to avoid circular dependency
        from items import get_all_items
        all_items = await get_all_items()
        item_ids = [item["value"] for item in all_items]
        await fetch_all_price_data(item_ids)
        return {"status": "success", "message": "Rankings refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh rankings: {e}")
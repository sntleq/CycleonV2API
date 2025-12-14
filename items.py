import asyncio
import csv
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter(prefix="/items")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE = CACHE_DIR / "item_details.json"
ITEMS_CSV = Path("datasets/Runescape_Item_Names.csv")
BASE_URL = "https://secure.runescape.com/m=itemdb_rs/api/catalogue/detail.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}

CONCURRENCY_LIMIT = 4
RETRIES = 5
RETRY_DELAY = 1  # seconds

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# Load cache into memory at startup
if CACHE_FILE.exists():
    with CACHE_FILE.open("r", encoding="utf-8") as f:
        ITEM_CACHE = json.load(f)
else:
    ITEM_CACHE = {}


async def fetch_item_from_api(client: httpx.AsyncClient, item_id: str):
    """Fetch raw item data from RuneScape API (no caching)."""
    url = f"{BASE_URL}?item={item_id}"
    last_error = None

    async with semaphore:
        for attempt in range(RETRIES):
            try:
                r = await client.get(url, headers=HEADERS)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                if not r.content:
                    raise RuntimeError("Empty response")

                data = r.json()
                item = data.get("item")
                if not item:
                    raise RuntimeError(f"No item field in response for {item_id}")

                return item

            except Exception as e:
                last_error = e
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"Failed after {RETRIES} retries for {item_id}: {last_error}")


async def fetch_item(client: httpx.AsyncClient, item_id: str):
    """Fetch item data with caching."""
    if item_id in ITEM_CACHE:
        return ITEM_CACHE[item_id]

    item = await fetch_item_from_api(client, item_id)

    # Save raw item to cache
    ITEM_CACHE[item_id] = item
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(ITEM_CACHE, f, ensure_ascii=False, indent=2)

    return item


@router.get("/{item_id}")
async def get_item(item_id: str):
    """Get a single item's filtered data (uses cache if available)."""
    try:
        async with httpx.AsyncClient() as client:
            item = await fetch_item(client, item_id)

        # Filter columns for API response
        filtered_item = {
            "id": item.get("id"),
            "name": item.get("name"),
            "description": item.get("description"),
            "icon": item.get("icon"),
            "type": item.get("type"),
        }
        return filtered_item

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("")
async def get_all_items():
    """Return all items from CSV as { value, label } (id → value, name → label)."""
    try:
        with ITEMS_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            items = [
                {"value": row["Name_ID"].strip(), "label": row["Name"].strip()}
                for row in reader
                if row["Name_ID"].strip()
            ]
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

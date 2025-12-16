from pathlib import Path
import shutil
import json
import requests
from datetime import datetime, timezone

CACHE_DIR = Path("cache")
PRICES_DIR = CACHE_DIR / "prices"


def clear_cache():
    if not CACHE_DIR.exists():
        print("Cache directory does not exist.")
        return

    for item in CACHE_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print("Cache directory cleared.")


def get_latest_timestamp_from_cache():
    """Get the latest timestamp from the first file in cache/prices/"""
    if not PRICES_DIR.exists() or not any(PRICES_DIR.iterdir()):
        return None

    # Get the first file in the prices directory
    first_file = next(PRICES_DIR.iterdir(), None)
    if not first_file or not first_file.is_file():
        return None

    try:
        with open(first_file, 'r') as f:
            data = json.load(f)
            if data and len(data) > 0:
                # Get the last item's timestamp
                return data[-1]['timestamp']
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error reading cache file: {e}")
        return None

    return None


def get_api_timestamp():
    """Get the latest timestamp from the API"""
    try:
        response = requests.get("https://api.weirdgloop.org/exchange", timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse the "rs" timestamp and convert to milliseconds
        rs_timestamp_str = data.get('rs')
        if rs_timestamp_str:
            dt = datetime.fromisoformat(rs_timestamp_str.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)  # Convert to milliseconds
    except Exception as e:
        print(f"Error fetching API timestamp: {e}")
        return None

    return None


def is_10am_utc():
    """Check if current time is 10 AM UTC"""
    now = datetime.now(timezone.utc)
    return now.hour == 10


def should_clear_cache():
    """Determine if cache should be cleared"""
    cache_timestamp = get_latest_timestamp_from_cache()

    # If cache/prices/ doesn't exist or is empty
    if cache_timestamp is None:
        if is_10am_utc():
            print("Cache is empty and it's 10 AM UTC. Clearing cache.")
            return True
        else:
            print("Cache is empty but it's not 10 AM UTC. Skipping cache clear.")
            return False

    # If cache exists, compare with API
    api_timestamp = get_api_timestamp()
    if api_timestamp is None:
        print("Could not fetch API timestamp. Skipping cache clear.")
        return False

    if cache_timestamp < api_timestamp:
        print(f"Cache is behind (cache: {cache_timestamp}, api: {api_timestamp}). Clearing cache.")
        return True
    else:
        print(f"Cache is up to date (cache: {cache_timestamp}, api: {api_timestamp}). Skipping cache clear.")
        return False


if __name__ == "__main__":
    if should_clear_cache():
        clear_cache()
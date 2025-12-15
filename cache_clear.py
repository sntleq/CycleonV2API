from pathlib import Path
import shutil

CACHE_DIR = Path("cache")


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


if __name__ == "__main__":
    clear_cache()

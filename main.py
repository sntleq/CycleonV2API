from fastapi import FastAPI
import items, predictions

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model before accepting requests
    print("Pre-loading TimesFM model on startup...")
    from predictions import get_model  # Replace with actual import
    get_model()
    print("Model loaded! Ready to accept requests.")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.include_router(items.router)
app.include_router(predictions.router)

from fastapi import FastAPI
import items, predictions

app = FastAPI()
app.include_router(items.router)
app.include_router(predictions.router)

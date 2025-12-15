from fastapi import FastAPI
import items, predictions, rankings

app = FastAPI()
app.include_router(items.router)
app.include_router(predictions.router)
app.include_router(rankings.router)
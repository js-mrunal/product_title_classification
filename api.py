from fastapi import FastAPI
import uvicorn
import os

from classfier_api.category_prediction_api import router as category_router

app = FastAPI(title="Product Category Predictor")
app.include_router(category_router)


if __name__ == "__main__":
    uvicorn.run(app="api:app", reload=True)
from fastapi import FastAPI

from classfier_api.category_prediction_api import router as category_router

app = FastAPI(title="Product Category Predictor")
app.include_router(category_router)

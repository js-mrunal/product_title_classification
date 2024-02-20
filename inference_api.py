from http.client import responses
from fastapi import FastAPI
from pydantic import BaseModel, Field
import gunicorn
from typing import List

class CategoryPredictionIn(BaseModel):
    product_titles : List[str] = Field(..., max_items=10)

app = FastAPI()

@app.post("/category/prediction/")
async def predict_category(titles: CategoryPredictionIn) -> CategoryPredictionIn:
    return titles.product_titles[:2]

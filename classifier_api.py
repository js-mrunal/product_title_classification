from asyncio import streams
from http.client import responses
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

import pandas as pd
from inference import DNNInference

class CategoryPredictionIn(BaseModel):
    product_titles : List[str] = Field(..., max_items=10)

app = FastAPI()

@app.post("/category/prediction/")
async def predict_category(titles: CategoryPredictionIn) -> dict:
    product_titles_df = pd.DataFrame(titles.product_titles, columns=['product_titles'])
    dnn_inference = DNNInference(
        feature_column = "product_titles",
        data = product_titles_df,
        save_dir_path= "model_data"
    )
    predicted_labels = dnn_inference.predict()
    return predicted_labels
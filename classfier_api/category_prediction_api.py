from typing import List

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field
import os
from inference.inference import DNNInference

router = APIRouter(prefix="/category", tags=["category"])


class CategoryPredictionIn(BaseModel):
    product_titles: List[str] = Field(..., max_items=10)


@router.post("/prediction")
async def predict_category(titles: CategoryPredictionIn) -> dict:
    product_titles_df = pd.DataFrame(titles.product_titles, columns=["product_titles"])
    dnn_inference = DNNInference(
        feature_column="product_titles",
        data=product_titles_df,
        save_dir_path=str(os.environ.get('APP_HOME', "."))+"/model_data",
    )
    predicted_labels = dnn_inference.predict()
    return predicted_labels

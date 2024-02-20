import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import DataLoader
from model import MulticlassDNN

import sys
sys.path.append('../')

def dnn_training(file_path: str) -> bool:
    # model training
    multiclass_nn = MulticlassDNN(
        file_path=file_path,
        feature_columns=["product_title"],
        label_column="category",
        save_dir_path= "../model_data"
    )

    multiclass_nn.plot_model()
    history = multiclass_nn.fit()

    # TODO: dump results
    return True


if __name__ == "__main__":
    file_path = "../data/gpc_product_titles_subset_data.csv"
    dnn_training(file_path=file_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import DataLoader
from model import MulticlassDNN


def dnn_training(file_path: str) -> bool:
    # model training
    multiclass_nn = MulticlassDNN(
        file_path=file_path,
        feature_column="product_title",
        label_column="category",
    )

    multiclass_nn.plot_model()
    history = multiclass_nn.fit()

    # todo: dump results
    return True


if __name__ == "__main__":
    file_path = "data/gpc_product_titles_subset_data.csv"
    dnn_training(file_path=file_path)

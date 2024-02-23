"""
Main script for training a neural network model on the specified data file.
"""

from model import MulticlassDNN


def dnn_training(file_path: str) -> bool:
    """
    Creates an instance of MulticlassDNN and trains a neural network
    on data read from the file_path.
    """
    multiclass_nn = MulticlassDNN(
        file_path=file_path,
        feature_columns=["product_title"],
        label_column="category",
        save_dir_path="./model_data",
    )

    multiclass_nn.plot_model()
    multiclass_nn.compile_fit()
    return True


if __name__ == "__main__":
    # Example Usage
    FILE_PATH = "data/gpc_product_titles_subset_data.csv"
    dnn_training(file_path=FILE_PATH)

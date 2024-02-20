from cProfile import label
from tkinter import Y
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from dataclasses import dataclass, field
from dataloader import DataLoader
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import argmax
from sklearn.metrics import accuracy_score, precision_score, recall_score

import sys
sys.path.append('../')
from utils.utils import save_pickle, save_model

@dataclass
class MulticlassDNN:
    file_path: str
    save_dir_path: str
    feature_columns: List[str]
    label_column: str

    train_x: pd.DataFrame = field(init=False)
    train_y: pd.DataFrame = field(init=False)
    test_x: pd.DataFrame = field(init=False)
    test_y: pd.DataFrame = field(init=False)

    feature_transformer: TfidfVectorizer = field(init=False)
    label_transformer: OneHotEncoder = field(init=False)

    model: keras.models.Model = field(init=False)

    def __post_init__(self):
        data_loader = DataLoader(file_path=self.file_path)
        product_titles_df = data_loader.get_data()
        train_df, test_df = data_loader.get_train_test_data()
        # prepare data for passing to DNN
        self.feature_transformer = self._build_feature_transformer(
            input_data=product_titles_df[self.feature_columns[0]]
        )
        self.label_transformer = self._build_label_transformer(
            output_data=product_titles_df[self.label_column]
        )

        self.train_x, self.train_y = self.format_data_for_nn(data_df=train_df)
        self.test_x, self.test_y = self.format_data_for_nn(data_df=test_df)

        # build DNN
        self.model = self._build_dnn(
            input_shape=(self.train_x.shape[1],), num_classes=self.train_y.shape[1]
        )

    def _build_dnn(self, input_shape: Tuple[int,], num_classes:int) -> keras.models.Model:
        input_layer = keras.layers.Input(shape=input_shape)
        y = keras.layers.Dense(2048)(input_layer)
        y = keras.layers.Dense(512)(y)
        output_layer = keras.layers.Dense(num_classes, activation="softmax")(y)
        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def _build_feature_transformer(self, input_data: pd.DataFrame) -> TfidfVectorizer:
        return TfidfVectorizer(min_df=5, stop_words="english").fit(input_data)

    def _build_label_transformer(self, output_data: pd.DataFrame) -> OneHotEncoder:
        return OneHotEncoder().fit(np.array(output_data).reshape(-1, 1))

    def format_data_for_nn(self, data_df: pd.DataFrame):
        x = self.feature_transformer.transform(data_df[self.feature_columns[0]]).toarray()
        y = self.label_transformer.transform(
            np.array(data_df[self.label_column]).reshape(-1, 1)
        ).todense()
        return x, y

    def get_model(self):
        return self.model

    def plot_model(self):
        # defensive check
        # since we are using post-init to define model,
        # we do not have to worry about model not existing
        assert self.model, "Model missing !"

        return tf.keras.utils.plot_model(
            self.model, show_shapes=True, show_layer_names=True
        )

    def fit(self):
        assert self.model, "Model missing !"

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = self.model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.20,
            epochs=1,
            batch_size=128,
            shuffle=True,
            verbose=True,
        )

        y_pred = argmax(self.model.predict(self.test_x), axis=1)
        y_true = argmax(self.test_y, axis=1)
        test_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        print("Accuracy on testing data: ",  test_acc)

        # save model weights
        if not os.path.exists(f"{self.save_dir_path}"):
            os.mkdir(f"{self.save_dir_path}")
        save_pickle(self.feature_transformer, f"{self.save_dir_path}/feature_transformer.pkl")
        save_pickle(self.label_transformer, f"{self.save_dir_path}/label_transformer.pkl")
        save_model(self.model, f"{self.save_dir_path}/classifier")

        return True
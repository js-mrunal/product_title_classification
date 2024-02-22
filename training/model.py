'''
This script ochestrates entire pipeline from data loading and preprocessing,
feature transformation to defining neural network and training it on the processed data. 
'''

import os
from dataclasses import dataclass, field
from typing import Tuple, List

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_pickle, save_model

import numpy as np
from numpy import argmax
import pandas as pd
import keras
import tensorflow as tf

from dataloader import DataLoader

@dataclass
class MulticlassDNN:
    '''
    Class for defining, compiling and fitting neural network

    Attributes:
    file_path: Path to the data file.
    save_dir_path: Path to save model architecture and weights.
    feature_columns: Feature column names
    label_column: Target column names
    
    '''
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
        '''
        Post initialization function to perform following operations.
        1. Data Loading.
        2. Splitting into train-test subsets.
        3. Transforming input and targets to feed to the neural network.
        4. Defining neural network architecture.
        '''
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
        '''
        Define neural network architecture.

        Parameters:
        input_shape: Tuple representing shape of the input data
        num_classes: Number of output units

        Returns:
        Defined neural network model.
        '''
        input_layer = keras.layers.Input(shape=input_shape)
        y = keras.layers.Dense(2048)(input_layer)
        y = keras.layers.Dense(512)(y)
        output_layer = keras.layers.Dense(num_classes, activation="softmax")(y)
        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def _build_feature_transformer(self, input_data: pd.DataFrame) -> TfidfVectorizer:
        '''
        Build a TD-IDF vectorizer on the data
        '''
        return TfidfVectorizer(min_df=5, stop_words="english").fit(input_data)

    def _build_label_transformer(self, output_data: pd.DataFrame) -> OneHotEncoder:
        '''
        Build One-Hot Encoder on the output labels
        '''
        return OneHotEncoder().fit(np.array(output_data).reshape(-1, 1))

    def format_data_for_nn(self, data_df: pd.DataFrame):
        '''
        Transforms input data and labels using defined TF-IDF and One-Hot Encoder respectively.
        '''
        x = self.feature_transformer.transform(data_df[self.feature_columns[0]]).toarray()
        y = self.label_transformer.transform(
            np.array(data_df[self.label_column]).reshape(-1, 1)
        ).todense()
        return x, y

    def get_model(self) -> keras.models.Model:
        '''
        Get the defined neural network model.
        '''
        return self.model

    def plot_model(self):
        '''
        Plot neural network architecture.
        '''
        # defensive check
        # since we are using post-init to define model,
        # we do not have to worry about model not existing
        assert self.model, "Model missing !"

        return tf.keras.utils.plot_model(
            self.model, show_shapes=True, show_layer_names=True
        )

    def compile_fit(self):
        '''
        Compile and fit neural network.

        Returns:
        None
        '''
        assert self.model, "Model missing !"

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
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

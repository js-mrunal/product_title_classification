import numpy as np
import pandas as pd
import keras
import tensorflow as tf


class MulticlassNN:
    def __init__(self, train_features, train_y, test_features, test_y):
        self.train_x = train_features
        self.train_y = train_y
        self.test_x = test_features
        self.test_y = test_y
        self.model = None

    def get_model(self, input_size=0, output_size=0):
        assert input_size > 0
        assert output_size > 0

        input_layer = keras.layers.Input(shape=(input_size,))
        y = keras.layers.Dense(2048)(input_layer)
        y = keras.layers.Dense(512)(y)
        output_layer = keras.layers.Dense(output_size, activation="softmax")(y)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def plot_model(self):
        if self.model != None:
            return tf.keras.utils.plot_model(
                self.model, show_shapes=True, show_layer_names=True
            )
        print("Model not defined.")

    def train_model(self):
        if self.model != None:
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            history = self.model.fit(
                self.train_x,
                self.train_y,
                validation_split=0.20,
                epochs=20,
                batch_size=128,
                shuffle=True,
                verbose=True,
            )
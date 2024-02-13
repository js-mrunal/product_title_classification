import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class MulticlassNN:
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_x: pd.DataFrame
    test_y: pd.DataFrame
    model: keras.models.Model = field(init = False)

    def __post_init__(self):
        input_layer = keras.layers.Input(shape=(self.train_x.shape[1],))
        y = keras.layers.Dense(2048)(input_layer)
        y = keras.layers.Dense(512)(y)
        output_layer = keras.layers.Dense(self.train_y.shape[1], activation="softmax")(y)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

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
        return self.model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.20,
            epochs=20,
            batch_size=128,
            shuffle=True,
            verbose=True,
        )
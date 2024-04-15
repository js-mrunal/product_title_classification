import pickle
import tensorflow as tf

def save_pickle(file_data, save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(file_data, fp)

def load_pickle(save_path):
    with open(save_path, "rb") as fp:
        return pickle.load(fp)

def save_model(dnn, save_path):
    dnn.save(f"{save_path}_model.keras")

def load_model(save_path:str):
    loaded_model = tf.keras.models.load_model(f"{save_path}_model.keras")
    return loaded_model

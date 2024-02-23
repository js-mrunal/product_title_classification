import pickle

from keras.models import model_from_json


def save_pickle(file_data, save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(file_data, fp)


def load_pickle(save_path):
    with open(save_path, "rb") as fp:
        return pickle.load(fp)


def save_model(dnn, save_path):
    # serialize model to JSON
    model_json = dnn.to_json()
    with open(save_path + "_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    dnn.save_weights(save_path + "_model.h5")


def load_model(save_path):
    # load json and create model
    json_file = open(save_path + "_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(save_path + "_model.h5")
    return loaded_model

import string
from dataclasses import dataclass, field
from functools import cached_property

import nltk
import pandas as pd
import keras
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from utils import load_model, load_pickle
nltk.download("stopwords")

@dataclass
class DNNInference:
    feature_column: str
    data: pd.DataFrame
    save_dir_path: str

    transformed_data: pd.DataFrame = field(init=False)
    feature_transformer: TfidfVectorizer = field(init=False)
    label_transformer: OneHotEncoder = field(init=False)
    model: keras.models.Model = field(init=False)

    @cached_property
    def stop_words(self):
        return stopwords.words("english")

    def __post_init__(self):
        try:
            self.model = load_model(f"{self.save_dir_path}/classifier")
            self.feature_transformer = load_pickle(
                f"{self.save_dir_path}/feature_transformer.pkl"
            )
            self.label_transformer = load_pickle(
                f"{self.save_dir_path}/label_transformer.pkl"
            )
        except Exception as exc:
            raise ValueError("Error in reading models and data files.") from exc
        self.data = self._preprocess_data(self.data)
        self.transformed_data = self.feature_transformer.transform(
            self.data[self.feature_column]
        ).toarray()

    def _preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # preprocess product title
        raw_data[self.feature_column] = raw_data[self.feature_column].apply(
            self._title_preprocessing
        )
        raw_data["word_count"] = raw_data[self.feature_column].str.split().str.len()
        return raw_data.loc[raw_data["word_count"] > 1]

    # defining this as a private function
    # as we will only use it internally
    def _title_preprocessing(self, text: str) -> str:
        text = text.replace("-", " ").strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        word_lst = [word for word in text.split() if word not in self.stop_words]
        return " ".join(word_lst)

    def predict(self):
        predicted_array = self.model.predict(self.transformed_data)
        return self.inverse_transform(predicted_array)

    
    def inverse_transform(self, y_pred):
        y_pred = self.label_transformer.inverse_transform(
            y_pred
        ).reshape(1, -1)[0]
        predictions = dict(
            zip(list(self.data[self.feature_column]), list(y_pred))
        )
        return predictions

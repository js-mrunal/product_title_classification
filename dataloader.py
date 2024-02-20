from cgi import test
import pandas as pd
import nltk
from typing import Tuple
from nltk.corpus import stopwords
import string

nltk.download("stopwords")
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from functools import cached_property


@dataclass
class DataLoader:
    file_path: str
    # initialising rest of the class variables
    data: pd.DataFrame = field(init=False)
    train_data: pd.DataFrame = field(init=False)
    test_data: pd.DataFrame = field(init=False)

    @cached_property
    def stop_words(self):
        return stopwords.words("english")

    def __post_init__(self):
        try:
            self.data = pd.read_csv(
                self.file_path,
                encoding="utf-8",
            )
            print("Data Loaded with shape: ", self.data.shape)
        except:
             raise ValueError("Error in reading data file")
        self.data = self._preprocess_data(self.data)
        self.train_data, self.test_data = self._train_test_split(p_test_size=0.10)

    def get_data(self):
        return self.data

    def get_train_test_data(self):
        return self.train_data, self.test_data

    def _preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # preprocess product title
        raw_data["product_title"] = raw_data["product_title"].apply(
            self._title_preprocessing
        )
        raw_data["word_count"] = raw_data["product_title"].str.split().str.len()
        return raw_data.loc[raw_data["word_count"] > 1]

    # defining this as a private function
    # as we will only use it internally
    def _title_preprocessing(self, text: str) -> str:
        text =  text.replace("-", " ").strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        word_lst = [word for word in text.split() if word not in self.stop_words]
        return " ".join(word_lst)

    def _train_test_split(
        self, p_shuffle=True, p_test_size=0.10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(self.data, shuffle=p_shuffle, test_size=p_test_size)

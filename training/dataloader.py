"""
Script for handling all data-related operations.
"""

import string
from dataclasses import dataclass, field
from functools import cached_property
from typing import Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")


@dataclass
class DataLoader:
    """
    A data loader that handles reading, preprocessing and train-test splitting

    Attributes:
    file_path: Path to the data file.

    """

    file_path: str
    # initialising rest of the class variables
    data: pd.DataFrame = field(init=False)
    train_data: pd.DataFrame = field(init=False)
    validate_data: pd.DataFrame = field(init=False)
    test_data: pd.DataFrame = field(init=False)

    @cached_property
    def stop_words(self):
        return stopwords.words("english")

    def __post_init__(self):
        """
        Reads data from the user-provided path.
        Preprocesses the product titles to remove stop words and special characters.
        Splits the entire dataset into train and test split.

        Assigns:
                                self.data: preprocessed dataset
        self.train_data: training dataset
        self.test_data: tested dataset
        """
        try:
            self.data = pd.read_csv(
                self.file_path,
                encoding="utf-8",
            )
            print("Data Loaded with shape: ", self.data.shape)
        except Exception as exc:
            raise ValueError("Error in reading data file") from exc
        self.data = self._preprocess_data(self.data)
        self.train_data, self.validate_data, self.test_data = (
            self._train_validate_test_split(p_test_size=0.10)
        )

    def get_data(self):
        """
        Returns entire preprocessed data read from the user-provided path.
        """
        return self.data

    def get_train_validate_test_data(self):
        """
        Returns train-test datasets which are already preprocessed.
        """
        return self.train_data, self.validate_data, self.test_data

    def _preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parent function for preprocessing product title text and adding word count.

        Parameters:
        raw_data: data read from the source

        Returns:
        Data with preprocessed product titles having word_count > 1
        """
        raw_data["product_title"] = raw_data["product_title"].apply(
            self._title_preprocessing
        )
        raw_data["word_count"] = raw_data["product_title"].str.split().str.len()
        return raw_data.loc[raw_data["word_count"] > 1]

    def _title_preprocessing(self, text: str) -> str:
        """
        Removes leading and trailing whitespaces in the text.
        Changes all text characters to lower case.
        Removes punctuations and stop words.

        Parameters:
        text: raw data string

        Returns:
        preprocessed text
        """
        text = text.replace("-", " ").strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        word_lst = [word for word in text.split() if word not in self.stop_words]
        return " ".join(word_lst)

    def _train_validate_test_split(
        self, p_shuffle=True, p_test_size=0.10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits preprocessed data into train and test subsets.

        Parameters:
        p_shuffle: whether or not to shuffle the data before splitting.
        p_test_size: Represents proportion of dataset to include in the test split.

        Returns:
        Dataframes containing train-test split of input.
        """
        train, test_df = train_test_split(
            self.data, shuffle=p_shuffle, test_size=p_test_size
        )
        train_df, validate_df = train_test_split(
            train, shuffle=p_shuffle, test_size=p_test_size
        )
        return train_df, validate_df, test_df

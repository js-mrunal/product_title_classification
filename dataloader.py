import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
nltk.download("stopwords")
from sklearn.model_selection import train_test_split

class DataHandling:
    def __init__(self, file_path=None, shortlisted_categories=None):
        self.file_path = file_path
        self.data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.STOP_WORDS = stopwords.words("english")
        self.shortlisted_categories = shortlisted_categories

    def read_data(self):
        try:
            self.data = pd.read_csv(
                self.file_path,
                encoding="utf-8",
                sep="\t",
            )
            print("Data Loaded with shape: ", self.data.shape)
        except:
            print("Error in reading data file")

    def preprocess_data(self):
        assert len(self.data) > 0, "No data loaded. Use read_data() first"
        gpc_data_categories = self.data.gpc_categories.str.split(">", expand=True)
        self.data = pd.concat(
            [self.data["product_title_cleaned"], gpc_data_categories[0]], axis=1
        )
        self.data = self.data.rename(
            columns={0: "category", "product_title_cleaned": "product_title"}
        )
        if self.shortlisted_categories != None:
            self.data = self.data.loc[
                self.data.category.isin(self.shortlisted_categories)
            ]
            print("\nTruncated data: ", self.data.shape)

        # preprocess product title
        self.data["product_title"] = self.data["product_title"].apply(
            self.title_preprocessing
        )
        self.data["word_count"] = self.data["product_title"].str.split().str.len()
        self.data = self.data.loc[self.data["word_count"] > 1]

    def title_preprocessing(self, text):
        if not isinstance(text, str):
            return ""

        text = text.strip().lower()
        text = text.replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
        word_lst = [word for word in text.split() if word not in self.STOP_WORDS]
        return " ".join(word_lst)

    def train_test_split(self, p_shuffle=True, p_test_size=0.10):
        assert len(self.data) > 0, "No data loaded. Use read_data() first"
        self.train_data, self.test_data = train_test_split(
            self.data, shuffle=p_shuffle, test_size=p_test_size
        )
        print(
            "Train data: ", self.train_data.shape, " Test data: ", self.test_data.shape
        )
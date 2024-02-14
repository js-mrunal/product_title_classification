import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from dataloader import DataHandling
from model import MulticlassNN


# read data
file_path = "data/gpc_product_titles_gemini_responses_cleaned.tsv"
shortlisted_categories = [
    "animals & pet supplies",
    "apparel & accessories",
    "electronics",
    "food, beverages & tobacco",
    "home & garden",
    #     "sporting goods",
]
gpc = DataHandling(file_path=file_path, shortlisted_categories=shortlisted_categories)

# prepare training features
tfidf = TfidfVectorizer(min_df=5, stop_words="english").fit(gpc.data.product_title)
train_x = tfidf.transform(gpc.train_data.product_title).toarray()
test_x = tfidf.transform(gpc.test_data.product_title).toarray()

# prepare prediction features
le = OneHotEncoder()
train_y = le.fit_transform(np.array(gpc.train_data.category).reshape(-1, 1)).todense()
test_y = le.transform(np.array(gpc.test_data.category).reshape(-1, 1)).todense()
print("Train Test shapes: ", train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# model training 
multiclass_nn = MulticlassNN(train_x, train_y, test_x, test_y)
multiclass_nn.plot_model()
history = multiclass_nn.fit()
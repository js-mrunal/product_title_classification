import pandas as pd
from inference import DNNInference
import sys
sys.path.append('../')

if __name__ == '__main__':

    test_product_titles = {'product_title': 
    ['industrial broom head heavy duty cleaning applications',
    'leon paul apex 2 sabre cuffs left hand',
    'led desk lamp wireless charging pad',
    'tapioca flour',
    'julienne peeler create culinary masterpieces perfect julienne strips']}

    test_product_titles = pd.DataFrame(test_product_titles)

    dnn_inference = DNNInference(
        feature_column = "product_title",
        data = test_product_titles,
        save_dir_path= "../model_data"
    )

    predicted_labels = dnn_inference.predict()
    print(predicted_labels)
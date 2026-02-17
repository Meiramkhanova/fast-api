import pickle
from pprint import pprint
import pandas as pd

class TranscriptPreprocessingPipeline:
    def __init__(self):
        # если знаешь, какие у него атрибуты
        self.attr1 = None
        self.attr2 = None

    # можно добавить методы, если нужно
    def info(self):
        return f"Pipeline с attr1={self.attr1}, attr2={self.attr2}"


file_path = "./transcript_preprocessing_pipeline.pkl"


with open(file_path, "rb") as file:
    data = pickle.load(file)

pprint(vars(data))

print("//////////////////////////////////////////")

with open("./item_cf_model.pkl", "rb") as file:
    data2 = pickle.load(file)

pprint(vars(data2))

print("//////////////////////////////////////////")

with open("./item_cf_strict_model.pkl", "rb") as file:
    data3 = pickle.load(file)

pprint(vars(data3))

print("//////////////////////////////////////////")

with open("./user_cf_model.pkl", "rb") as file:
    data4 = pickle.load(file)

pprint(vars(data4))




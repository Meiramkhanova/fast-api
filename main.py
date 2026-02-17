import pickle
from pprint import pprint

class TranscriptPreprocessingPipeline:
    def __init__(self):
        # если знаешь, какие у него атрибуты
        self.attr1 = None
        self.attr2 = None

    # можно добавить методы, если нужно
    def info(self):
        return f"Pipeline с attr1={self.attr1}, attr2={self.attr2}"


file_path = "C:/Users/adina/Downloads/transcript_preprocessing_pipeline.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

print(type(data))
pprint(vars(data))  # показывает атрибуты объекта

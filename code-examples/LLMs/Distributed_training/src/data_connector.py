from datasets import load_dataset

class DataConnector:

    @staticmethod
    def get_data(path):
        return load_dataset(path)
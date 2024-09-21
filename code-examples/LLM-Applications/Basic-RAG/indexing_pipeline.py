# load data
# transform data
# index data

from indexing.data_loading import DataLoader
from indexing.data_transform import DataProcessor
from indexing.data_indexing import DataIndexer

FILE = '../book/ESLII.pdf'

def run():

    loader = DataLoader(FILE)
    processor = DataProcessor()
    indexer = DataIndexer()

    # load data
    docs = loader.get_docs()
    # transform data
    transformed_docs = processor.transform(docs)
    # index data
    indexer.index_data(transformed_docs)

if __name__ == '__main__':
    run()



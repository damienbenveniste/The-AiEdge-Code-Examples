from indexing.text_splitter import TextSplitter
from indexing.graph_extraction import GraphExtractor
from indexing.graph_resolving import GraphResolver
from indexing.data_indexing import DataIndexer
from indexing.communities import CommunitySummarizer

def run():

    text_splitter = TextSplitter()
    graph_extractor = GraphExtractor()
    graph_resolver = GraphResolver()
    data_indexer = DataIndexer()
    summarizer = CommunitySummarizer()

    nodes = text_splitter.load_data()
    nodes = graph_extractor.extract(nodes)
    entities, relationships = graph_resolver.resolve(nodes)
    summarizer.run(entities, relationships)
    data_indexer.insert_data(entities, relationships)

if __name__ == "__main__":
    run()
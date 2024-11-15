from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


class TextSplitter:

    def load_data(self, directory="./../book"):
        documents = SimpleDirectoryReader(directory).load_data()
        node_parser = SentenceSplitter(chunk_size=1200, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes
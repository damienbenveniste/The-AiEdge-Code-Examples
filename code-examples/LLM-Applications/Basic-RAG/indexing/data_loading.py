
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataLoader:

    def __init__(self, file_path, chunk_size=20000, chunk_overlap=100):

        self.loader = PyPDFLoader(file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_docs(self):
        return self.loader.load_and_split(
            text_splitter=self.text_splitter
    )

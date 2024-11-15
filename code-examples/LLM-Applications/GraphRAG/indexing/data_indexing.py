from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from openai import OpenAI
from data_models import KeywordsModel
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


system_prompt = """
Given some initial query, generate synonyms or related keywords up to 10 in total, considering possible cases of pluralization, common expressions, etc.
The resulting list should be a list of entity names used to index a graph database.
"""


class DataIndexer:

    def __init__(self):
        self.graph_store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="llamaindex",
            url="bolt://localhost:7687",
        )

    def get_embeddings(self, texts: list[str]):

        data = client.embeddings.create(
            input = texts, 
            model="text-embedding-3-small"
        ).data
        embeddings = [d.embedding for d in data]
        return embeddings
    
    def vector_search(self, query: str, similarity_top_k=10):
        embedding = self.get_embeddings(query)[0]
        query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=similarity_top_k
        )
        nodes = self.graph_store.vector_query(query)[0]
        return nodes
    
    def get_synonyms(self, query):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUERY: {query}"},
            ],
            response_format=KeywordsModel,
        )
        keywords_model = completion.choices[0].message.parsed
        keywords = [k.capitalize() for k in keywords_model.keywords]
        return keywords

    def keyword_search(self, query: str):
        keywords = self.get_synonyms(query)
        nodes = self.graph_store.get(ids=keywords)
        return nodes
    
    def get_related_nodes(self, nodes):
        triplets = self.graph_store.get_rel_map(nodes)
        nodes = []
        for triplet in triplets:
            nodes.append(triplet[0])
            nodes.append(triplet[-1])

        return nodes

    def retrieve(self, query):
        nodes_from_vector = self.vector_search(query)
        nodes_from_keywords = self.keyword_search(query)
        nodes = self.get_related_nodes(nodes_from_vector + nodes_from_keywords)
        nodes_dict = {n.name: n for n in nodes}
        return list(nodes_dict.values())

    def insert_data(self, entities, relationships):

        texts_index = [str(entity) for entity in entities]
        embeddings = self.get_embeddings(texts_index)

        for entity, embedding in zip(entities, embeddings):
            entity.embedding = embedding

        self.graph_store.upsert_nodes(entities)
        self.graph_store.upsert_relations(relationships)

        # refresh schema if needed
        if self.graph_store.supports_structured_queries:
            self.graph_store.get_schema(refresh=True)


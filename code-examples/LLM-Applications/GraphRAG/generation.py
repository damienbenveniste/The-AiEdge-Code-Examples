import nest_asyncio
from openai import OpenAI
import os

nest_asyncio.apply()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

system_prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Only provide the answer and nothing else!
"""

class Generator:

    def __init__(self, data_indexer, community_summarizer):
        self.indexer = data_indexer
        self.summarizer = community_summarizer

    def get_entities(self, query):
        entities = self.indexer.retrieve(query)
        return entities

    def get_community_summaries(self, query):
        entities = self.get_entities(query)
        all_summaries = set()
        for entity in entities:
            summaries = self.summarizer.get_summaries_for_entity(entity.name)
            all_summaries.update(summaries)

        return all_summaries
    
    def generate(self, query):
        summaries = self.get_community_summaries(query)
        context = "\n\n".join(summaries)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT: {context}\n\nQUERY: {query}"},
            ],
        )

        return response.choices[0].message.content
    

if __name__ == '__main__':
    from indexing.communities import CommunitySummarizer
    from indexing.data_indexing import DataIndexer

    summarizer = CommunitySummarizer()
    indexer = DataIndexer()
    summarizer.load()

    generator = Generator(indexer, summarizer)

    response = generator.generate('Who are the main characters of the book?')
    print(response)
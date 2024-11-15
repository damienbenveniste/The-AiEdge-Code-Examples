import networkx as nx
from graspologic.partition import hierarchical_leiden
from openai import OpenAI
from collections import defaultdict
import pickle
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

system_prompt = """
You are provided with a set of entities and their relationships from a knowledge graph.
The entities are represented as `entity->entity_type->description` and the relationships as `entity1->entity2->relation->relationship_description.`
Your task is to create a summary of this community of entities and relationships. 
The summary should include the names of the entities involved and a concise synthesis of the relationship descriptions. 
The goal is to capture the most critical and relevant details that highlight the nature and significance of each relationship. 
Ensure that the summary is coherent and integrates the information in a way that emphasizes the key aspects of the relationships."
"""

class CommunitySummarizer:

    def __init__(self):
        self.summaries_dict = None
        self.community_dict = None

    def create_nx_graph(self, relationships):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for relationship in relationships:
            nx_graph.add_node(relationship.source_id)
            nx_graph.add_node(relationship.target_id)
            nx_graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relationship=relationship.label,
                description=relationship.properties["relationship_description"],
            )
        return nx_graph
    
    def create_communities(self, nx_graph):
        return hierarchical_leiden(nx_graph, max_cluster_size=5)
    
    def get_communities(self, clusters, entities, relationships):
        entity_dict = defaultdict(list)
        relationship_dict = defaultdict(list)
        self.community_dict = defaultdict(list)
        for cluster in clusters:
            for entity in entities:
                if cluster.node == entity.name:
                    entity_dict[cluster.cluster].append(entity)

            for relationship in relationships:
                if cluster.node == relationship.source_id or cluster.node == relationship.target_id:
                    relationship_dict[cluster.cluster].append(relationship)

            self.community_dict[cluster.node].append(cluster.cluster)
        
        return entity_dict, relationship_dict
    
    def summarize_communities(self, entity_dict, relationship_dict):
        summaries_dict = {}
        for cluster, entities in entity_dict.items():
            relationships = relationship_dict[cluster]
            summary = self.summarize_community(entities, relationships)
            summaries_dict[cluster] = summary

        return summaries_dict
    
    def get_summaries_for_entity(self, entity_name):
        if not (self.summaries_dict and self.community_dict):
            raise Exception('Missing summaries')
        
        if entity_name not in self.community_dict:
            return []
        
        communities = self.community_dict[entity_name]
        summaries = [self.summaries_dict[c] for c in communities]
        return summaries

    def summarize_community(self, entities, relationships):

        entities_text = "\n".join([f"{e.name}->{e.label}->{e.properties['entity_description']}" for e in entities])
        relationships_text = "\n".join([f"{r.source_id}->{r.target_id}->{r.label}->{r.properties["relationship_description"]}" for r in relationships])

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"entities: {entities_text}\n\nrelationships: {relationships_text}"},
            ],
        )

        return completion.choices[0].message.content
    
    def save(self, file_name='communities.pkl'):
        with open(file_name, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
    def load(self, file_name='communities.pkl'):
        with open(file_name, 'rb') as inp:
            obj = pickle.load(inp)
            self.__dict__.update(obj.__dict__)
        return self
       
    def run(self, entities, relationships):
        nx_graph = self.create_nx_graph(relationships)
        clusters = self.create_communities(nx_graph)
        entity_dict, relationship_dict = self.get_communities(clusters, entities, relationships)
        self.summaries_dict = self.summarize_communities(entity_dict, relationship_dict)
        self.save()






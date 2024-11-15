from openai import OpenAI
from llama_index.core.schema import TextNode
from data_models import KnowledgeModel
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from multiprocessing import Pool, cpu_count
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

system_prompt = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
"""

class GraphExtractor:

    def extract_from_node(self, node: TextNode):

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"text: {node}"}
            ],
            response_format=KnowledgeModel,
        )

        knowledge_model = completion.choices[0].message.parsed
        entities, relationships = self.convert_to_llamaindex(knowledge_model)
        node.metadata[KG_NODES_KEY] = entities
        node.metadata[KG_RELATIONS_KEY] = relationships 
        return node
    
    def convert_to_llamaindex(self, knowledge_model: KnowledgeModel):
        entities = []
        relationships = []

        for entity_model in knowledge_model.entities:
            entity = EntityNode(
                name=entity_model.name, 
                label=entity_model.type, 
                properties={"entity_description": entity_model.description}
            )
            entities.append(entity)

        valid_entities = {entity.name for entity in entities}

        for relationship_model in knowledge_model.relationships:
            if not (relationship_model.source_entity.name in valid_entities and relationship_model.target_entity.name in valid_entities):
                pass
            relationship = Relation(
                label=relationship_model.relation,
                source_id=relationship_model.source_entity.name,
                target_id=relationship_model.target_entity.name,
                properties={"relationship_description": relationship_model.description}
            )
            relationships.append(relationship)
        
        return entities, relationships
    
    def extract(self, nodes):
        with Pool(cpu_count()) as pool:
            nodes = pool.map(self.extract_from_node, nodes)
        return nodes
    



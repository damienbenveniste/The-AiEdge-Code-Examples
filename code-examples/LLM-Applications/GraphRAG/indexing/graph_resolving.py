
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation
)
from openai import OpenAI
from collections import defaultdict
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


system_prompt = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
"""


class GraphResolver:

    def summarize_entity(self, descriptions, entity_name):

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"entity: {entity_name}\n\ndescriptions: {descriptions}"},
            ]
        )
        return completion.choices[0].message.content
    
    def summarize_relation(self, descriptions, source_entity, target_entity, relation):

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Source_entity: {source_entity}\nTarget_entity: {target_entity}\nRelation: {relation}\n\ndescriptions: {descriptions}"},
            ]
        )
        return completion.choices[0].message.content
    
    def resolve_entities(self, nodes):
        entities = []
        for node in nodes:
            entities.extend(node.metadata[KG_NODES_KEY])

        entities_dict = defaultdict(list)
        for entity in entities:
            entities_dict[entity.name].append(entity)

        final_entities = []

        for name, entities in entities_dict.items():
            if len(entities) == 1:
                description = entities[0].properties["entity_description"]
            else:
                descriptions = "\n\n".join([node.properties["entity_description"] for node in entities])
                description = self.summarize_entity(descriptions, name)
            final_entity = EntityNode(
                name=name, 
                label=entities[0].label, 
                properties={"entity_description": description}
            )
            final_entities.append(final_entity)

        return final_entities
    
    def resolve_relationships(self, nodes):
        relationships = []

        for node in nodes:
            relationships.extend(node.metadata[KG_RELATIONS_KEY])

        relationships_dict = defaultdict(list)
        for relationship in relationships:
            key = (relationship.source_id, relationship.target_id, relationship.label)
            relationships_dict[key].append(relationship)

        final_relationships = []

        for (source_entity, target_entity, relation), relationships in relationships_dict.items():
            if len(relationships) == 1:
                description = relationships[0].properties["relationship_description"]
            else:
                descriptions = "\n\n".join([node.properties["relationship_description"] for node in relationships])
                description = self.summarize_relation(descriptions, source_entity, target_entity, relation)
            final_relationship = Relation(
                label=relation,
                source_id=source_entity,
                target_id=target_entity,
                properties={"relationship_description": description}
            )
            final_relationships.append(final_relationship)

        return final_relationships
    
    def resolve(self, nodes):
        entities = self.resolve_entities(nodes)
        relationships = self.resolve_relationships(nodes)

        return entities, relationships


            






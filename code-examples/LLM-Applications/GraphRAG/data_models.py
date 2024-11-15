from pydantic import BaseModel, Field, field_validator
from enum import Enum

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    OTHER = "OTHER"


class EntityModel(BaseModel):
    """
    This model represents an entity extracted from a text chunk
    """
    name: str = Field(
        description="Name of the entity, capitalized"
    )
    type: EntityType = Field(
        description="Type of the entity",
        default=EntityType.OTHER
    )
    description: str = Field(
        description="Comprehensive description of the entity's attributes and activities"
    )
    @field_validator('name')
    def capitalize_name(cls, value: str) -> str:
        return value.capitalize()
    

class RelationshipModel(BaseModel):
    """
    This model represents an relationship between 2 entities extracted from a text chunk
    """
    source_entity: EntityModel = Field(
        description="Source entity"
    )
    target_entity: EntityModel = Field(
        description="Target entity"
    )
    relation: str = Field(
        description="relationship between source_entity and target_entity"
    )
    description: str = Field(
        description="explanation as to why you think the source entity and the target entity are related to each other"
    )


class KnowledgeModel(BaseModel):
    """
    Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
    """
    entities: list[EntityModel] = Field(
        description="Identify all entities"
    )
    relationships: list[RelationshipModel] = Field(
        description="Identify all pairs of (source_entity, target_entity) that are *clearly related* to each other."
    )


class KeywordsModel(BaseModel):
    """
    This model represents the list or synonyms or related keywords related to the user query
    """
    keywords: list[str] = Field(
        description="synonyms or related keywords used as a list of entity names to index a graph database"
    )
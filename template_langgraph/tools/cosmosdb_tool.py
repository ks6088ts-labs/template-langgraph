from functools import lru_cache

from azure.cosmos import CosmosClient, PartitionKey
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class Settings(BaseSettings):
    cosmosdb_host: str = "<AZURE_COSMOS_DB_ENDPOINT>"
    cosmosdb_key: str = "<AZURE_COSMOS_DB_KEY>"
    cosmosdb_database_name: str = "template_langgraph"
    cosmosdb_container_name: str = "kabuto"
    cosmosdb_partition_key: str = "/id"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_cosmosdb_settings() -> Settings:
    """Get Cosmos DB settings."""
    return Settings()


class CosmosdbClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_cosmosdb_settings()
        self.vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=CosmosClient(
                url=settings.cosmosdb_host,
                credential=settings.cosmosdb_key,
            ),
            embedding=AzureOpenAiWrapper().embedding_model,
            vector_embedding_policy={
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "distanceFunction": "cosine",
                        "dimensions": 1536,
                    }
                ]
            },
            indexing_policy={
                "indexingMode": "consistent",
                "includedPaths": [
                    {"path": "/*"},
                ],
                "excludedPaths": [
                    {"path": '/"_etag"/?'},
                ],
                "vectorIndexes": [
                    {"path": "/embedding", "type": "diskANN"},
                ],
                "fullTextIndexes": [
                    {"path": "/text"},
                ],
            },
            cosmos_container_properties={
                "partition_key": PartitionKey(path=settings.cosmosdb_partition_key),
            },
            cosmos_database_properties={},
            full_text_policy={
                "defaultLanguage": "en-US",
                "fullTextPaths": [
                    {
                        "path": "/text",
                        "language": "en-US",
                    }
                ],
            },
            database_name=settings.cosmosdb_database_name,
            container_name=settings.cosmosdb_container_name,
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Add documents to a Cosmos DB container."""
        return self.vector_store.add_documents(
            documents=documents,
        )

    def delete_documents(
        self,
        ids: list[str],
    ) -> bool | None:
        """Delete documents from a Cosmos DB container."""
        return self.vector_store.delete(
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """Perform a similarity search in the Cosmos DB index."""
        return self.vector_store.similarity_search(
            query=query,
            k=k,  # Number of results to return
        )


class CosmosdbInput(BaseModel):
    query: str = Field(
        default="禅モード",
        description="Query to search in the Cosmos DB index",
    )
    k: int = Field(
        default=5,
        description="Number of results to return from the similarity search",
    )


class CosmosdbOutput(BaseModel):
    content: str = Field(description="Content of the document")
    id: str = Field(description="ID of the document")


@tool(args_schema=CosmosdbInput)
def search_cosmosdb(query: str, k: int = 5) -> list[CosmosdbOutput]:
    """Search for similar documents in CosmosDB vector store.

    Args:
        query: The search query string
        k: Number of results to return (default: 5)

    Returns:
        CosmosdbOutput: A Pydantic model containing the search results
    """
    wrapper = CosmosdbClientWrapper()
    documents = wrapper.similarity_search(
        query=query,
        k=k,
    )
    outputs = []
    for document in documents:
        outputs.append(
            {
                "content": document.page_content,
                "id": document.id,
            }
        )
    return outputs

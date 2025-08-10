from functools import lru_cache

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class Settings(BaseSettings):
    ai_search_key: str = "<your-ai-search-key>"
    ai_search_endpoint: str = "<your-ai-search-endpoint>"
    ai_search_index_name: str = "<your-ai-index-name>"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_ai_search_settings() -> Settings:
    """Get AI Search settings."""
    return Settings()


class AiSearchClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_ai_search_settings()
        self.vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=settings.ai_search_endpoint,
            azure_search_key=settings.ai_search_key,
            index_name=settings.ai_search_index_name,
            embedding_function=AzureOpenAiWrapper().embedding_model.embed_query,
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Add documents to a Cosmos DB container."""
        return self.vector_store.add_documents(
            documents=documents,
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

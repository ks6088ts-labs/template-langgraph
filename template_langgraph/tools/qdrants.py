from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http.models import UpdateResult
from qdrant_client.models import Distance, PointStruct, VectorParams


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_qdrant_settings() -> Settings:
    """Get Qdrant settings."""
    return Settings()


class QdrantClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_qdrant_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
    ) -> bool:
        """Create a collection in Qdrant."""
        result = self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        return result

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> UpdateResult:
        """Upsert points into a Qdrant collection."""
        return self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

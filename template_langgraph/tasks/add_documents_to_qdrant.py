import logging

from qdrant_client.models import PointStruct

from template_langgraph.loggers import get_logger
from template_langgraph.models.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.csv_loaders import CsvLoaderWrapper
from template_langgraph.tools.qdrants import QdrantClientWrapper

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
COLLECTION_NAME = "documents"

if __name__ == "__main__":
    # Load documents from CSV files
    documents = CsvLoaderWrapper().load_csv_docs()
    logger.info(f"Loaded {len(documents)} documents from CSV.")

    # hardcoded collection name for demonstration purposes

    logger.info(f"Upserting {len(documents)} documents into Qdrant collection: {COLLECTION_NAME}")
    points = []
    for i, doc in enumerate(documents):
        logger.debug(f"Processing document {i}: {doc.metadata.get('source', 'unknown')}")
        content = doc.page_content
        content = content.replace(" ", "")
        embedding = AzureOpenAiWrapper().create_embedding(content)
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "file_name": doc.metadata.get("source", f"doc_{i}"),
                    "content": content,
                },
            )
        )

    qdrant_client = QdrantClientWrapper()
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vector_size=len(points[0].vector) if points else 1536,  # default vector size
    )

    logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    operation_info = qdrant_client.upsert_points(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    logger.info(f"Upserted {len(points)} points into Qdrant collection: {COLLECTION_NAME}")

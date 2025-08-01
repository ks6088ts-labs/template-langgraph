import logging

from template_langgraph.loggers import get_logger
from template_langgraph.tools.qdrants import QdrantClientWrapper

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
COLLECTION_NAME = "qa_kabuto"

if __name__ == "__main__":
    logger.info(f"Deleting Qdrant collection: {COLLECTION_NAME}")
    result = QdrantClientWrapper().delete_collection(
        collection_name=COLLECTION_NAME,
    )
    if result:
        logger.info(f"Successfully deleted Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.warning(f"Qdrant collection {COLLECTION_NAME} does not exist or could not be deleted.")
    logger.info("Deletion task completed.")

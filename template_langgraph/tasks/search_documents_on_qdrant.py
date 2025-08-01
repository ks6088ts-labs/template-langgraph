import logging

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.qdrants import QdrantClientWrapper

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
COLLECTION_NAME = "qa_kabuto"

if __name__ == "__main__":
    question = "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。"
    qdrant_client = QdrantClientWrapper()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=AzureOpenAiWrapper().create_embedding(question),
    )
    logger.info(f"Found {len(results)} results for the question: {question}")
    for result in results:
        logger.info(f"File Name: {result.payload['file_name']}")
        logger.info(f"Content: {result.payload['content']}")
        logger.info("-" * 40)

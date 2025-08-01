import logging

from template_langgraph.loggers import get_logger
from template_langgraph.tools.elasticsearch_tool import ElasticsearchClientWrapper

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
COLLECTION_NAME = "docs_kabuto"

if __name__ == "__main__":
    query = "禅モード"
    es = ElasticsearchClientWrapper()

    results = es.search(
        index_name=COLLECTION_NAME,
        query=query,
    )
    logger.info(f"Found {len(results)} results for the question: {query}")
    for i, result in enumerate(results, start=1):
        logger.info(f"Result {i}:")
        logger.info(f"File Name: {result.metadata['source']}")
        logger.info(f"Content: {result.page_content}")
        logger.info("-" * 40)

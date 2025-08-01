import logging

from template_langgraph.loggers import get_logger
from template_langgraph.tools.elasticsearch_tool import ElasticsearchClientWrapper
from template_langgraph.tools.pdf_loaders import PdfLoaderWrapper

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
COLLECTION_NAME = "docs_kabuto"

if __name__ == "__main__":
    # Create Elasticsearch index
    es = ElasticsearchClientWrapper()
    logger.info(f"Creating Elasticsearch index: {COLLECTION_NAME}")
    result = es.create_index(
        index_name=COLLECTION_NAME,
    )
    if result:
        logger.info(f"Created Elasticsearch index: {COLLECTION_NAME}")
    else:
        logger.warning(f"Index {COLLECTION_NAME} already exists.")

    # Load documents from PDF files
    documents = PdfLoaderWrapper().load_pdf_docs()
    logger.info(f"Loaded {len(documents)} documents from PDF.")

    # Add documents to Elasticsearch index
    result = es.add_documents(
        index_name=COLLECTION_NAME,
        documents=documents,
    )
    if result:
        logger.info(f"Added {len(documents)} documents to Elasticsearch index: {COLLECTION_NAME}")
    else:
        logger.error(f"Failed to add documents to Elasticsearch index: {COLLECTION_NAME}")

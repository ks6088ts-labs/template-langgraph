import logging

import typer
from dotenv import load_dotenv

from template_langgraph.internals.pdf_loaders import PdfLoaderWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.elasticsearch_tool import ElasticsearchClientWrapper

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Elasticsearch operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def search_documents(
    index_name: str = typer.Option(
        "docs_kabuto",
        "--index-name",
        "-i",
        help="Name of the Elasticsearch index to search documents in",
    ),
    query: str = typer.Option(
        "禅モード",
        "--query",
        "-q",
        help="Query to search in the Elasticsearch index",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    es = ElasticsearchClientWrapper()

    results = es.search(
        index_name=index_name,
        query=query,
    )
    logger.info(f"Found {len(results)} results for the question: {query}")
    for i, result in enumerate(results, start=1):
        logger.info(f"Result {i}:")
        logger.info(f"File Name: {result.metadata['source']}")
        logger.info(f"Content: {result.page_content}")
        logger.info("-" * 40)


@app.command()
def create_index(
    index_name: str = typer.Option(
        "docs_kabuto",
        "--index-name",
        "-i",
        help="Name of the Elasticsearch index to create",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    es = ElasticsearchClientWrapper()
    logger.info(f"Creating Elasticsearch index: {index_name}")
    result = es.create_index(
        index_name=index_name,
    )
    if result:
        logger.info(f"Created Elasticsearch index: {index_name}")
    else:
        logger.warning(f"Index {index_name} already exists.")


@app.command()
def delete_index(
    index_name: str = typer.Option(
        "docs_kabuto",
        "--index-name",
        "-i",
        help="Name of the Elasticsearch index to delete",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    es = ElasticsearchClientWrapper()
    logger.info(f"Deleting Elasticsearch index: {index_name}")
    result = es.delete_index(
        index_name=index_name,
    )
    if result:
        logger.info(f"Deleted Elasticsearch index: {index_name}")
    else:
        logger.warning(f"Index {index_name} does not exist or could not be deleted.")
        return False


@app.command()
def add_documents(
    index_name: str = typer.Option(
        "docs_kabuto",
        "--index-name",
        "-i",
        help="Name of the Elasticsearch index to add documents to",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Create Elasticsearch index
    es = ElasticsearchClientWrapper()

    # Load documents from PDF files
    documents = PdfLoaderWrapper().load_pdf_docs()
    logger.info(f"Loaded {len(documents)} documents from PDF.")

    # Add documents to Elasticsearch index
    result = es.add_documents(
        index_name=index_name,
        documents=documents,
    )
    if result:
        logger.info(f"Added {len(documents)} documents to Elasticsearch index: {index_name}")
    else:
        logger.error(f"Failed to add documents to Elasticsearch index: {index_name}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()

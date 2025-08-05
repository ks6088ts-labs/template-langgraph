import logging

import typer
from dotenv import load_dotenv

from template_langgraph.loggers import get_logger
from template_langgraph.tools.cosmosdb_tool import CosmosdbClientWrapper
from template_langgraph.utilities.csv_loaders import CsvLoaderWrapper
from template_langgraph.utilities.pdf_loaders import PdfLoaderWrapper

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Cosmos DB operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def add_documents(
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

    # Load documents from PDF files
    pdf_documents = PdfLoaderWrapper().load_pdf_docs()
    logger.info(f"Loaded {len(pdf_documents)} documents from PDF.")

    # Load documents from CSV files
    csv_documents = CsvLoaderWrapper().load_csv_docs()
    logger.info(f"Loaded {len(csv_documents)} documents from CSV.")

    # Combine all documents
    documents = pdf_documents + csv_documents
    logger.info(f"Total documents to add: {len(documents)}")

    # Add documents to Cosmos DB
    cosmosdb_client = CosmosdbClientWrapper()
    ids = cosmosdb_client.add_documents(
        documents=documents,
    )
    logger.info(f"Added {len(ids)} documents to Cosmos DB.")
    for id in ids:
        logger.debug(f"Added document ID: {id}")

    # assert cosmosdb_client.delete_documents(ids=ids), "Failed to delete documents from Cosmos DB"


@app.command()
def similarity_search(
    query: str = typer.Option(
        "禅モード",
        "--query",
        "-q",
        help="Query to search in the Cosmos DB index",
    ),
    k: int = typer.Option(
        5,
        "--k",
        "-k",
        help="Number of results to return from the similarity search",
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

    logger.info(f"Searching Cosmos DB with query: {query}")

    # Perform similarity search
    cosmosdb_client = CosmosdbClientWrapper()
    documents = cosmosdb_client.similarity_search(
        query=query,
        k=k,  # Number of results to return
    )
    logger.info(f"Found {len(documents)} results for query: {query}")

    # Log the results
    for i, document in enumerate(documents, start=1):
        logger.debug("-" * 40)
        logger.debug(f"#{i}: {document.model_dump_json(indent=2)}")


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()

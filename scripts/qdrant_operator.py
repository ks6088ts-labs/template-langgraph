import logging

import typer
from dotenv import load_dotenv
from qdrant_client.models import PointStruct

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.loggers import get_logger
from template_langgraph.tools.qdrants import QdrantClientWrapper
from template_langgraph.utilities.csv_loaders import CsvLoaderWrapper

# Initialize the Typer application
app = typer.Typer(
    add_completion=False,
    help="Qdrant operator CLI",
)

# Set up logging
logger = get_logger(__name__)


@app.command()
def delete_collection(
    collection_name: str = typer.Option(
        "qa_kabuto",
        "--collection-name",
        "-c",
        help="Name of the Qdrant collection to delete",
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

    logger.info(f"Deleting Qdrant collection: {collection_name}")
    result = QdrantClientWrapper().delete_collection(
        collection_name=collection_name,
    )
    if result:
        logger.info(f"Successfully deleted Qdrant collection: {collection_name}")
    else:
        logger.warning(f"Qdrant collection {collection_name} does not exist or could not be deleted.")
    logger.info("Deletion task completed.")


@app.command()
def add_documents(
    collection_name: str = typer.Option(
        "qa_kabuto",
        "--collection-name",
        "-c",
        help="Name of the Qdrant collection to add documents to",
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

    # Load documents from CSV files
    documents = CsvLoaderWrapper().load_csv_docs()
    logger.info(f"Loaded {len(documents)} documents from CSV.")

    points = []
    embedding_wrapper = AzureOpenAiWrapper()
    for i, doc in enumerate(documents):
        logger.debug(f"Processing document {i}: {doc.metadata.get('source', 'unknown')}")
        content = doc.page_content.replace("\n", " ")
        logger.debug(f"Creating embedding for document {i} with content: {content[:50]}...")
        vector = embedding_wrapper.create_embedding(content)
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "file_name": doc.metadata.get("source", f"doc_{i}"),
                    "content": content,
                },
            )
        )

    # Create Qdrant collection and upsert points
    logger.info(f"Creating Qdrant collection: {collection_name}")
    qdrant_client = QdrantClientWrapper()
    qdrant_client.create_collection(
        collection_name=collection_name,
        vector_size=len(points[0].vector) if points else 1536,  # default vector size
    )

    # Upsert points into the Qdrant collection
    logger.info(f"Upserting points into Qdrant collection: {collection_name}")
    operation_info = qdrant_client.upsert_points(
        collection_name=collection_name,
        points=points,
    )
    logger.info(f"Upserted {len(points)} points into Qdrant collection: {collection_name}")
    logger.info(f"Operation info: {operation_info}")


@app.command()
def search_documents(
    collection_name: str = typer.Option(
        "qa_kabuto",
        "--collection-name",
        "-c",
        help="Name of the Qdrant collection to search documents in",
    ),
    question: str = typer.Option(
        "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。",
        "--question",
        "-q",
        help="Question to search in the Qdrant collection",
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

    qdrant_client = QdrantClientWrapper()

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=AzureOpenAiWrapper().create_embedding(question),
    )
    logger.info(f"Found {len(results)} results for the question: {question}")
    for result in results:
        logger.info(f"File Name: {result.payload['file_name']}")
        logger.info(f"Content: {result.payload['content']}")
        logger.info("-" * 40)


if __name__ == "__main__":
    load_dotenv(
        override=True,
        verbose=True,
    )
    app()

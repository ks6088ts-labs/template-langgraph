from mcp.server.fastmcp import FastMCP

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    logger.info("Starting Math MCP server...")
    mcp.run(transport="stdio")

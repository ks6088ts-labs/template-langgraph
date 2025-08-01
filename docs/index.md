# template-langgraph

## Operations

```bash
# Delete collection from Qdrant
uv run python -m template_langgraph.tasks.delete_qdrant_collection

# Add documents to Qdrant
uv run python -m template_langgraph.tasks.add_documents_to_qdrant

# Search Qdrant
uv run python -m template_langgraph.tasks.search_documents_on_qdrant

# Run Kabuto Helpdesk Agent
uv run python -m template_langgraph.tasks.run_kabuto_helpdesk_agent "KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。"
```

## References

### Models

- [AzureOpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/azureopenai/)

### Tools

- [CSVLoader](https://python.langchain.com/docs/how_to/document_loader_csv/)
- [Qdrant](https://github.com/qdrant/qdrant)

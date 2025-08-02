# template-langgraph

## Operations

```bash
# Start Docker containers
docker compose up -d

# Delete collection from Qdrant
uv run python -m template_langgraph.tasks.delete_qdrant_collection

# Add documents to Qdrant
uv run python -m template_langgraph.tasks.add_documents_to_qdrant

# Search Qdrant
uv run python -m template_langgraph.tasks.search_documents_on_qdrant

# Add documents to Elasticsearch
uv run python -m template_langgraph.tasks.add_documents_to_elasticsearch

# Search Elasticsearch
uv run python -m template_langgraph.tasks.search_documents_on_elasticsearch

# Run Kabuto Helpdesk Agent
uv run python -m template_langgraph.tasks.run_kabuto_helpdesk_agent "KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。"
uv run python -m template_langgraph.tasks.run_kabuto_helpdesk_agent "KABUTOのマニュアルから禅モードに関する情報を教えて下さい"

# BasicWorkflowAgent
uv run python -m template_langgraph.tasks.draw_basic_workflow_agent_mermaid_png "data/basic_workflow_agent.png"
uv run python -m template_langgraph.tasks.run_basic_workflow_agent
# KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください
# 天狗のいたずら という現象について KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください

# IssueFormatterAgent
uv run python -m template_langgraph.tasks.run_issue_formatter_agent
# KABUTOにログインできない！パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。

```

## References

### LangGraph

- [Build a custom workflow](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

### Sample Codes

- [「現場で活用するためのAIエージェント実践入門」リポジトリ](https://github.com/masamasa59/genai-agent-advanced-book)

### Models

- [AzureOpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/azureopenai/)

### Tools

- [CSVLoader](https://python.langchain.com/docs/how_to/document_loader_csv/)
- [Qdrant](https://github.com/qdrant/qdrant)

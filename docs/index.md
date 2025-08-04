# template-langgraph

## Overview

```shell
# Set up environment
docker compose up -d
```

## Testing

see [test_all.sh](../scripts/test_all.sh) for all operations

```shell
# Test all scripts
bash scripts/test_all.sh
```

## References

### LangGraph

- [Build a custom workflow](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [LangGraphの(LLMなし)Human-in-the-loopを試してみた](https://qiita.com/te_yama/items/db38201af60dec76384d)
- [🤖 LangGraph Multi-Agent Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)
- [Software Design誌「実践LLMアプリケーション開発」第24回サンプルコード](https://github.com/mahm/softwaredesign-llm-application/tree/main/24)

### Sample Codes

- [「現場で活用するためのAIエージェント実践入門」リポジトリ](https://github.com/masamasa59/genai-agent-advanced-book)

### Models

- [AzureOpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/azureopenai/)

### Tools

- [CSVLoader](https://python.langchain.com/docs/how_to/document_loader_csv/)
- [Qdrant](https://github.com/qdrant/qdrant)

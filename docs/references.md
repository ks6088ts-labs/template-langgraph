## References

### LangGraph

- [Build a custom workflow](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [LangGraph の(LLM なし)Human-in-the-loop を試してみた](https://qiita.com/te_yama/items/db38201af60dec76384d)
- [🤖 LangGraph Multi-Agent Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)
- [Software Design 誌「実践 LLM アプリケーション開発」第 24 回サンプルコード](https://github.com/mahm/softwaredesign-llm-application/tree/main/24)
- [Streamlit](https://python.langchain.com/docs/integrations/callbacks/streamlit/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
  - [Research Agent with MCP Integration.](https://github.com/langchain-ai/deep_research_from_scratch/blob/main/src/deep_research_from_scratch/research_agent_mcp.py)
- [Command: A new tool for building multi-agent architectures in LangGraph](https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/)
- [Combine control flow and state updates with Command](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#combine-control-flow-and-state-updates-with-command)
- [Command: a new tool for building multi-agent architectures in LangGraph](https://www.youtube.com/watch?v=6BJDKf90L9A)
- [masamasa59/genai-agent-advanced-book > chapter6](https://github.com/masamasa59/genai-agent-advanced-book/blob/main/chapter6/arxiv_researcher/agent/paper_search_agent.py)
- [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- [Custom UI for Deep Agents](https://github.com/langchain-ai/deep-agents-ui)
- [How to deploy self-hosted standalone server](https://docs.langchain.com/langgraph-platform/deploy-standalone-server)
- [「現場で活用するための AI エージェント実践入門」リポジトリ](https://github.com/masamasa59/genai-agent-advanced-book)
- [Add and manage memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Chatbot with message summarization & external DB memory](https://github.com/langchain-ai/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb)
- [LangGraph の会話履歴を SQLite に保持しよう](https://www.creationline.com/tech-blog/chatgpt-ai/75797)

### LangChain

- [AzureOpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/azureopenai/)
- [CSVLoader](https://python.langchain.com/docs/how_to/document_loader_csv/)
- [Qdrant](https://github.com/qdrant/qdrant), [LangChain / Qdrant](https://python.langchain.com/docs/integrations/vectorstores/qdrant/)
- [Azure Cosmos DB No SQL](https://python.langchain.com/docs/integrations/vectorstores/azure_cosmos_db_no_sql/)
- [Azure AI Search](https://python.langchain.com/docs/integrations/vectorstores/azuresearch/)

### Azure AI Foundry

- [Quickstart: Get started with Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/quickstarts/get-started-code?tabs=python&pivots=fdp-project)
- [azure-rest-api-specs/specification/ai/data-plane/Azure.AI.Agents](https://github.com/Azure/azure-rest-api-specs/tree/main/specification/ai/data-plane/Azure.AI.Agents)
- [How to use the Deep Research tool](https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools/deep-research-samples?pivots=python)

#### Vision

- [What is Azure AI Vision?](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview)

### Services

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://python.langchain.com/docs/integrations/callbacks/streamlit/)

### Observability

- [OpenTelemetry](https://opentelemetry.io/)
  - [Python / Getting Started](https://opentelemetry.io/docs/languages/python/getting-started/)
  - [Python / Cookbook](https://opentelemetry.io/docs/languages/python/cookbook/)
- [OpenTelemetry Collector](https://opentelemetry.io//docs/collector/)
  - [OpenTelemetry Collector / Quick Start](https://opentelemetry.io/docs/collector/quick-start/)
  - [zPages / Exposed zPages routes](https://github.com/open-telemetry/opentelemetry-collector/blob/v0.132.0/extension/zpagesextension/README.md#exposed-zpages-routes)
- [Jaeger](https://www.jaegertracing.io/)
  - [Jaeger / Minimal deployment example (Elasticsearch backend)](https://www.jaegertracing.io/docs/1.72/deployment/#minimal-deployment-example-elasticsearch-backend)
- [Prometheus](https://prometheus.io/)
  - [Prometheus / Getting Started](https://prometheus.io/docs/prometheus/latest/getting_started/)
- [Trace with LangGraph](https://docs.langchain.com/langsmith/trace-with-langgraph)

### n8n

- [Hosting n8n / Installation / Server setups / Docker-Compose](https://docs.n8n.io/hosting/installation/server-setups/docker-compose/)

### Audio

- [Python の sounddevice を改めて試す](https://zenn.dev/kun432/scraps/f56760d41fc5aa)
- [How To Install libportaudio2 on Ubuntu 22.04](https://www.installati.one/install-libportaudio2-ubuntu-22-04/): `sudo apt-get -y install libportaudio2`
- [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)
- [python-soundfile](https://github.com/bastibe/python-soundfile)

### OpenAI

#### Realtime API

- [August 2025 / Realtime API audio model GA](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/whats-new#realtime-api-audio-model-ga)
- [Global Standard model availability](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions#global-standard-model-availability)
- [specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2025-04-01-preview/inference.json](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2025-04-01-preview/inference.json)
- [Realtime API with WebSocket](https://platform.openai.com/docs/guides/realtime-websocket)
- [GPT-4o Realtime API for speech and audio](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/realtime-audio-quickstart?tabs=keyless%2Clinux&pivots=programming-language-python)
- [OpenAI Python API library > examples/realtime](https://github.com/openai/openai-python/tree/main/examples/realtime)
- [How to use the GPT-4o Realtime API via WebRTC](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio-webrtc)

#### Responses API

- [OpenAI / New tools for building agents](https://openai.com/index/new-tools-for-building-agents/)
- [OpenAI / Responses](https://platform.openai.com/docs/api-reference/responses)
- [Azure OpenAI Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses?tabs=python-key)
- [LangChain / Responses API](https://python.langchain.com/docs/integrations/chat/openai/#responses-api)

### Hugging Face

- [LM Studio](https://lmstudio.ai/)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

### DSPy

- [DSPy (Declarative Self-improving Python)](https://dspy.ai/)
- [Language Models](https://dspy.ai/learn/programming/language_models/)
- [Language Models / v3.0.3](https://github.com/stanfordnlp/dspy/blob/3.0.3/docs/docs/learn/programming/language_models.md)
- [Software Design 誌「実践 LLM アプリケーション開発」第 25 回サンプルコード](https://github.com/mahm/softwaredesign-llm-application/tree/main/25)

### [MLflow](https://mlflow.org/docs/latest/genai/)

- [LangChain / MLflow](https://docs.langchain.com/oss/python/integrations/providers/mlflow_tracking)
- [MLflow / Tracing LangGraph🦜🕸️](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/)
- [GenAI Evaluation Quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/)

### [LiteLLM](https://docs.litellm.ai/)

- [Azure OpenAI](https://docs.litellm.ai/docs/providers/azure/)

### [Langfuse](https://langfuse.com/)

- [Cookbook: LangGraph Integration](https://langfuse.com/guides/cookbook/integration_langgraph)

### [Codex CLI](https://github.com/openai/codex)

- Azure OpenAI で Codex CLI を使う: [Codex Azure OpenAI Integration: Fast & Secure Code Development](https://devblogs.microsoft.com/all-things-azure/codex-azure-openai-integration-fast-secure-code-development/)
- [OpenAI Codex CLI のクイックスタート](https://note.com/npaka/n/n7b6448020250)

```shell
# Install Codex CLI
npm install -g @openai/codex

# Generate shell completion scripts
codex completion zsh

# Dump configurations
cat ~/.codex/config.toml

# Set up environment variables
export AZURE_OPENAI_API_KEY="<your-api-key>"

# MCP server management: https://qiita.com/tomada/items/2eb8d5b5173a4d70b287
## Add a global MCP server entry
codex mcp add context7   -- npx -y @upstash/context7-mcp
codex mcp add playwright -- npx -y @playwright/mcp@latest
codex mcp add mslearn -- npx -y mcp-remote "https://learn.microsoft.com/api/mcp" # ref. https://zenn.dev/yanskun/articles/codex-remote-mcp
## Remove MCP server
codex mcp remove context7
## List MCP servers
codex mcp list

# Run Codex non-interactively
codex exec "Playwright MCP を使って Yahoo リアルタイム検索の上位キーワードをまとめて"
```

### Model Context Protocol (MCP)

- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)

```shell
# Running the Inspector
npx @modelcontextprotocol/inspector

# Run the Codex MCP server (stdio transport)
codex mcp-server --help

# Run the Inspector with the Codex MCP server
npx @modelcontextprotocol/inspector -- codex mcp-server
```

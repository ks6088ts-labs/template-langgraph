# LangGraph AI Agent Template

A comprehensive template project for building AI agents using [LangGraph](https://langchain-ai.github.io/langgraph/), demonstrating various agent patterns, tool integration, and real-world use cases.

## What is LangGraph?

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework built on top of [LangChain](https://python.langchain.com/) that enables you to create stateful, multi-agent workflows. Unlike traditional chatbots that handle single interactions, LangGraph allows you to build complex AI systems that can:

- Maintain conversation state across multiple turns
- Use tools and external APIs
- Implement complex reasoning patterns
- Coordinate multiple AI agents
- Handle cyclical workflows and conditional logic

This template demonstrates these capabilities through practical examples using a fictional system called "KABUTO" for troubleshooting scenarios.

## Project Overview

This project showcases different AI agent patterns and architectures, from simple tool-calling agents to complex multi-agent systems. The examples use a fictional technical support scenario to demonstrate how agents can retrieve information from multiple data sources and provide structured responses.

### Why This Template Exists

Most AI applications need to:

1. **Access external information** - LLMs don't have access to your specific data
2. **Use tools** - Perform actions beyond text generation
3. **Maintain context** - Remember previous interactions
4. **Handle complex workflows** - Break down tasks into manageable steps

This template provides working examples of all these patterns using LangGraph.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Modern Python package manager
- [GNU Make](https://www.gnu.org/software/make/) - For running common tasks
- [Docker](https://www.docker.com/) - For running vector databases (optional)
- Azure
  - [Azure OpenAI](https://learn.microsoft.com/ja-jp/azure/ai-foundry/openai/overview) - LLM API
  - [Azure Cosmos DB](https://learn.microsoft.com/ja-jp/azure/cosmos-db/) - Data storage (optional)

## Quick Start

### 1. Environment Setup

```shell
# Clone the repository
git clone https://github.com/ks6088ts-labs/template-langgraph.git
cd template-langgraph

# Install Python dependencies
uv sync --all-extras

# Create environment configuration
cp .env.template .env
# Edit .env with your API keys (Azure OpenAI, etc.)
```

### 2. Start Supporting Services (Optional)

For full functionality, start the vector databases:

```shell
# Start Qdrant and Elasticsearch using Docker
docker compose up -d
```

### 3. Initialize Data Sources

**Set up Qdrant vector database:**

```shell
uv run python scripts/qdrant_operator.py add-documents \
  --collection-name qa_kabuto \
  --verbose
```

**Set up Elasticsearch search index:**

```shell
uv run python scripts/elasticsearch_operator.py create-index \
  --index-name docs_kabuto \
  --verbose
```

## Project Structure

### Core Components

- **`data/`** - Sample data for the fictional KABUTO system (PDFs, FAQs, troubleshooting guides)
- **`template_langgraph/`** - Main Python package containing all agent implementations
- **`notebooks/`** - Jupyter notebooks with interactive examples and explanations
- **`scripts/`** - Command-line tools for running agents

### Agent Examples (`template_langgraph/agents/`)

This project includes several agent implementations, each demonstrating different LangGraph patterns:

#### 1. `kabuto_helpdesk_agent/` - **Start Here!**

A simple agent using LangGraph's prebuilt `create_react_agent` function. This is the best starting point for understanding the basics.

**Key concepts:** ReAct pattern, tool calling, prebuilt agents

#### 2. `chat_with_tools_agent/` - **Core Implementation**

A manual implementation of the same logic as the helpdesk agent, showing how LangGraph workflows are built from scratch.

**Key concepts:** Graph construction, state management, node functions, edges

#### 3. `issue_formatter_agent/` - **Structured Output**

Demonstrates how to get structured data from AI responses using Pydantic models.

**Key concepts:** Structured output, data validation, response formatting

#### 4. `task_decomposer_agent/` - **Planning & Decomposition**

Shows how to break down complex tasks into smaller, manageable steps.

**Key concepts:** Task planning, multi-step reasoning, conditional workflows

#### 5. `supervisor_agent/` - **Multi-Agent Coordination**

Implements the supervisor pattern where one agent coordinates multiple specialized agents.

**Key concepts:** Multi-agent systems, agent coordination, supervisor patterns

### Supporting Modules

- **`template_langgraph/llms/`** - LLM API wrappers (Azure OpenAI, etc.)
- **`template_langgraph/tools/`** - Tool implementations for search, data retrieval
- **`template_langgraph/utilities/`** - Helper functions for document loading and processing

## Running the Examples

### Option 1: LangGraph Studio (Recommended for Development)

[LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) provides a visual interface for developing and debugging agents:

```shell
uv run langgraph dev
```

This opens a web interface where you can:

- Visualize agent workflows
- Step through executions
- Debug state transitions
- Test different inputs

![langgraph-studio.png](./images/langgraph-studio.png)

### Option 2: Jupyter Notebooks (Best for Learning)

Interactive notebooks with explanations and examples:

```shell
uv run jupyter lab
# Navigate to http://localhost:8888 and open notebooks/*.ipynb
```

![jupyterlab.png](./images/jupyterlab.png)

### Option 3: Command Line (Quick development)

Run agents from the terminal:

```shell
uv run python scripts/agent_operator.py run \
  --name "chat_with_tools_agent" \
  --question "KABUTO startup issue: screen flashes purple and system freezes" \
  --verbose
```

Example output showing the agent's reasoning process:

```text
Event: {'chat_with_tools': {'messages': [AIMessage(content='', tool_calls=[
  {'name': 'search_elasticsearch', 'args': {'keywords': 'KABUTO startup purple flashing freeze'}},
  {'name': 'search_qdrant', 'args': {'keywords': 'KABUTO startup purple flashing freeze'}}
])]}}

Event: {'tools': {'messages': [ToolMessage(content='Found documentation about startup protocol...')]}}

Event: {'chat_with_tools': {'messages': [AIMessage(content='
### Problem Analysis
The purple screen flashing during KABUTO startup indicates a "Shinobi Protocol" initialization failure...

### Solution
1. **Disconnect peripheral devices**: Unplug all connected devices for 5+ seconds
2. **Clear external clock cache**: This resolves clock synchronization issues
3. **Restart KABUTO**: Use the "Dragon Ball" button for 5+ seconds if needed
')]}}
```

### Option 4: FastAPI (Production-ready)

Run the FastAPI server to expose the agent as an API:

```shell
uv run fastapi run \
  --host "0.0.0.0" \
  --port 8000 \
  --workers 4 \
  template_langgraph/services/fastapis/main.py
# Access the API at http://localhost:8000/docs via Swagger UI
```

This allows you to interact with the agent programmatically via HTTP requests.

![fastapi.png](./images/fastapi.png)

## Key Concepts Demonstrated

### 1. **ReAct Pattern** (Reasoning + Acting)

The foundation of modern AI agents - the ability to reason about what to do, take actions, and reason about the results.

### 2. **Tool Calling**

How agents can use external functions to:

- Search databases (Elasticsearch, Qdrant)
- Call APIs
- Process files
- Execute calculations

### 3. **State Management**

How LangGraph maintains context across multiple interaction steps, enabling complex multi-turn conversations.

### 4. **Conditional Workflows**

Using graph structures to create branching logic based on agent decisions or external conditions.

### 5. **Multi-Agent Systems**

Coordinating multiple specialized agents to handle complex tasks that require different expertise.

## Data Sources Explained

The project uses fictional data about a system called "KABUTO" to demonstrate real-world scenarios:

- **`data/docs_kabuto.pdf`** - Technical documentation (simulates user manuals)
- **`data/qa_kabuto.csv`** - FAQ database (simulates past support tickets)
- **`data/docs_kabuto.md`** - Additional documentation

This fictional data serves a purpose: it proves that the AI agents can work with information that isn't in the LLM's training data, demonstrating the value of retrieval-augmented generation (RAG).

## Next Steps

1. **Start with the basics**: Run the `kabuto_helpdesk_agent` example
2. **Understand the implementation**: Compare it with `chat_with_tools_agent`
3. **Explore advanced patterns**: Try the task decomposer and supervisor agents
4. **Build your own**: Use this template as a starting point for your use case

## Learning Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

## Architecture Examples

This template demonstrates several proven agent architectures:

1. **Single Agent with Tools** - Basic tool-calling pattern
2. **ReAct Agent** - Reasoning and acting in loops
3. **Structured Output Agent** - Returning formatted data
4. **Planning Agent** - Breaking down complex tasks
5. **Supervisor Agent** - Coordinating multiple agents

Each pattern is implemented with clear examples and documentation to help you understand when and how to use them.

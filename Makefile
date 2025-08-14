# Git
GIT_REVISION ?= $(shell git rev-parse --short HEAD)
GIT_TAG ?= $(shell git describe --tags --abbrev=0 --always | sed -e s/v//g)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help

.PHONY: info
info: ## show information
	@echo "GIT_REVISION: $(GIT_REVISION)"
	@echo "GIT_TAG: $(GIT_TAG)"

.PHONY: install-deps-dev
install-deps-dev: ## install dependencies for development
	uv sync --all-extras
	uv run pre-commit install

.PHONY: install-deps
install-deps: ## install dependencies for production
	uv sync --no-dev

.PHONY: format-check
format-check: ## format check
	uv run ruff format --check --verbose

.PHONY: format
format: ## format code
	uv run ruff format --verbose

.PHONY: fix
fix: format ## apply auto-fixes
	uv run ruff check --fix

.PHONY: lint
lint: ## lint
	uv run ruff check .
	uv run ty check

.PHONY: test
test: ## run tests
	uv run pytest --capture=no -vv

.PHONY: ci-test
ci-test: install-deps-dev format-check lint test ## run CI tests

.PHONY: update
update: ## update packages
	uv lock --upgrade

.PHONY: jupyterlab
jupyterlab: ## run Jupyter Lab
	uv run jupyter lab

# ---
# Docker
# ---
DOCKER_REPO_NAME ?= ks6088ts
DOCKER_IMAGE_NAME ?= template-langgraph
DOCKER_COMMAND ?=

# Tools
TOOLS_DIR ?= /usr/local/bin
TRIVY_VERSION ?= 0.58.1

.PHONY: docker-build
docker-build: ## build Docker image
	docker build \
		-t $(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):$(GIT_TAG) \
		--build-arg GIT_REVISION=$(GIT_REVISION) \
		--build-arg GIT_TAG=$(GIT_TAG) \
		.

.PHONY: docker-run
docker-run: ## run Docker container
	docker run --rm $(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):$(GIT_TAG) $(DOCKER_COMMAND)

.PHONY: docker-lint
docker-lint: ## lint Dockerfile
	docker run --rm -i hadolint/hadolint < Dockerfile

.PHONY: docker-scan
docker-scan: ## scan Docker image
	@# https://aquasecurity.github.io/trivy/v0.18.3/installation/#install-script
	@which trivy || curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b $(TOOLS_DIR) v$(TRIVY_VERSION)
	trivy image $(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):$(GIT_TAG)

.PHONY: ci-test-docker
ci-test-docker: docker-lint docker-build docker-scan docker-run ## run CI test for Docker

# ---
# Docs
# ---

.PHONY: install-deps-docs
install-deps-docs: ## install dependencies for documentation
	uv sync --group docs

.PHONY: docs
docs: ## build documentation
	uv run mkdocs build

.PHONY: docs-serve
docs-serve: ## serve documentation
	uv run mkdocs serve

.PHONY: ci-test-docs
ci-test-docs: install-deps-docs docs ## run CI test for documentation

# ---
# Project
# ---

.PHONY: langgraph-studio
langgraph-studio: ## run LangGraph Studio
	uv run langgraph dev

.PHONY: fastapi-dev
fastapi-dev: ## run FastAPI
	uv run fastapi dev ./template_langgraph/services/fastapis/main.py

.PHONY: fastapi
fastapi: ## run FastAPI in production mode
	uv run fastapi run \
		--host "0.0.0.0" \
		--port 8000 \
		--workers 4 \
		template_langgraph/services/fastapis/main.py

.PHONY: streamlit
streamlit: ## run Streamlit
	uv run streamlit run \
		template_langgraph/services/streamlits/main.py

# ---
# Project / Create indices
# ---

.PHONY: create-cosmosdb-index
create-cosmosdb-index: ## create Cosmos DB index
	uv run python scripts/cosmosdb_operator.py --help
	uv run python scripts/cosmosdb_operator.py add-documents --verbose
	uv run python scripts/cosmosdb_operator.py similarity-search --query "禅モード" --k 3 --verbose

.PHONY: create-ai-search-index
create-ai-search-index: ## create Azure AI Search index
	uv run python scripts/ai_search_operator.py --help
	uv run python scripts/ai_search_operator.py add-documents --verbose
	uv run python scripts/ai_search_operator.py similarity-search --query "禅モード" --k 3 --verbose

COLLECTION_NAME ?= qa_kabuto
.PHONY: create-qdrant-index
create-qdrant-index: ## create Qdrant index
	uv run python scripts/qdrant_operator.py --help
	uv run python scripts/qdrant_operator.py delete-collection --collection-name $(COLLECTION_NAME) --verbose
	uv run python scripts/qdrant_operator.py add-documents     --collection-name $(COLLECTION_NAME) --verbose
	uv run python scripts/qdrant_operator.py search-documents  --collection-name $(COLLECTION_NAME) --verbose --question "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。"

INDEX_NAME ?= docs_kabuto
.PHONY: create-elasticsearch-index
create-elasticsearch-index: ## create Elasticsearch index
	uv run python scripts/elasticsearch_operator.py --help
	uv run python scripts/elasticsearch_operator.py delete-index     --index-name $(INDEX_NAME) --verbose
	uv run python scripts/elasticsearch_operator.py create-index     --index-name $(INDEX_NAME) --verbose
	uv run python scripts/elasticsearch_operator.py add-documents    --index-name $(INDEX_NAME) --verbose
	uv run python scripts/elasticsearch_operator.py search-documents --index-name $(INDEX_NAME) --verbose --query "禅モード"

# ---
# Project / Run agents
# ---
QUESTION ?= "KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください"
.PHONY: run-chat-with-tools-agent
run-chat-with-tools-agent: ## run chat with tools agent
	uv run python scripts/agent_operator.py run \
		--name chat_with_tools_agent \
		--question $(QUESTION) \
		--verbose

.PHONY: run-image-classifier-agent
run-image-classifier-agent: ## run image classifier agent
	uv run python scripts/agent_operator.py image-classifier-agent \
		--prompt "この画像の中身を 3 行で日本語で説明してください" \
		--file-paths "docs/images/fastapi.png,docs/images/streamlit.png" \
		--verbose

.PHONY: run-issue-formatter-agent
run-issue-formatter-agent: ## run issue formatter agent
	uv run python scripts/agent_operator.py run \
		--name issue_formatter_agent \
		--question "KABUTOにログインできない。パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。" \
		--verbose

.PHONY: run-kabuto-helpdesk-agent
run-kabuto-helpdesk-agent: ## run kabuto helpdesk agent
	uv run python scripts/agent_operator.py run \
		--name kabuto_helpdesk_agent \
		--question $(QUESTION) \
		--verbose

.PHONY: run-news-summarizer-agent
run-news-summarizer-agent: ## run news summarizer agent
	uv run python scripts/agent_operator.py news-summarizer-agent \
		--prompt "こちらの文章を 3 行で日本語で要約してください。" \
		--urls "https://raw.githubusercontent.com/ks6088ts-labs/template-langgraph/refs/heads/main/docs/index.md,https://raw.githubusercontent.com/ks6088ts-labs/template-langgraph/refs/heads/main/docs/deployment.md" \
		--verbose

.PHONY: run-task-decomposer-agent
run-task-decomposer-agent: ## run task decomposer agent
	uv run python scripts/agent_operator.py run \
		--name task_decomposer_agent \
		--question "KABUTOにログインできない。パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。" \
		--verbose

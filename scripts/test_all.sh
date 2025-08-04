#!/bin/bash

# Qdrant
uv run python scripts/qdrant_operator.py --help
uv run python scripts/qdrant_operator.py delete-collection --collection-name qa_kabuto --verbose
uv run python scripts/qdrant_operator.py add-documents --collection-name qa_kabuto --verbose
uv run python scripts/qdrant_operator.py search-documents --collection-name qa_kabuto --question "「鬼灯」を実行すると、KABUTOが急に停止します。原因と対策を教えてください。" --verbose

# Elasticsearch
uv run python scripts/elasticsearch_operator.py --help
uv run python scripts/elasticsearch_operator.py delete-index --index-name docs_kabuto --verbose
uv run python scripts/elasticsearch_operator.py create-index --index-name docs_kabuto --verbose
uv run python scripts/elasticsearch_operator.py add-documents --index-name docs_kabuto --verbose
uv run python scripts/elasticsearch_operator.py search-documents --index-name docs_kabuto --query "禅モード" --verbose

# Agents

## Draw agent graph
mkdir -p generated
AGENT_NAMES=(
    "chat_with_tools_agent"
    "issue_formatter_agent"
    "kabuto_helpdesk_agent"
    "task_decomposer_agent"
)
for AGENT_NAME in "${AGENT_NAMES[@]}"; do
    uv run python scripts/agent_runner.py png --name "$AGENT_NAME" --verbose --output "generated/${AGENT_NAME}.png"
done

## Run agents
# An array of pairs of agent names and questions
NAME_QUESTION_ARRAY=(
    "chat_with_tools_agent:KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください"
    "issue_formatter_agent:KABUTOにログインできない！パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。"
    "kabuto_helpdesk_agent:天狗のいたずら という現象について KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください"
    "task_decomposer_agent:KABUTOにログインできない！パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。"
)
for NAME_QUESTION in "${NAME_QUESTION_ARRAY[@]}"; do
    IFS=':' read -r AGENT_NAME QUESTION <<< "$NAME_QUESTION"
    echo "Running agent: $AGENT_NAME with question: $QUESTION"
    uv run python scripts/agent_runner.py run --name "$AGENT_NAME" --verbose --question "$QUESTION"
done

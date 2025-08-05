# template-langgraph

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [GNU Make](https://www.gnu.org/software/make/)

## Getting Started

### Set up local environment

```shell
# Clone the repository
git clone https://github.com/ks6088ts-labs/template-langgraph.git
cd template-langgraph

# Start services using Docker Compose (Requires Docker)
docker compose up -d

# Create a .env file based on the template
cp .env.template .env

# Install required Python packages (Requires uv)
uv sync --all-extras
```

### Set up infrastructure

Here are some commands you can use:

**Create Qdrant collection:**

> [!NOTE]
> Qdrant service is expected to be running

```shell
uv run python scripts/qdrant_operator.py add-documents \
  --collection-name qa_kabuto \
  --verbose
```

**Create Elasticsearch index:**

> [!NOTE]
> Elasticsearch service is expected to be running

```shell
uv run python scripts/elasticsearch_operator.py create-index \
  --index-name docs_kabuto \
  --verbose
```

### Run applications

#### LangGraph Studio

From the command line, you can run the LangGraph Studio to interact with the application.

```shell
uv run langgraph dev
```

![langgraph-studio.png](./images/langgraph-studio.png)

#### Jupyter Lab

From Jupyter Lab, you can run the notebooks in the `notebooks` directory to run various applications.

```shell
# Run Jupyter Lab
uv run jupyter lab

# Go to http://localhost:8888 in your browser and run notebooks/*.ipynb.
```

![jupyterlab.png](./images/jupyterlab.png)

#### Terminal

From the command line, you can run scripts in the `scripts` directory to run various applications.

```shell
uv run python scripts/agent_operator.py run \
  --name "chat_with_tools_agent" \
  --question "KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブル シュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください" \
  --verbose
2025-08-05 11:27:40,949 [    INFO] -------------------- (agent_operator.py:104)
2025-08-05 11:27:40,949 [    INFO] Event: {'chat_with_tools': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_LTenxsczZMuCdIAK8IBT9e7r', 'function': {'arguments': '{"keywords": "KABUTO 起動 紫色 点滅 フリーズ"}', 'name': 'search_elasticsearch'}, 'type': 'function'}, {'index': 1, 'id': 'call_POOYozcQOSjnXEaVkbxQpTz7', 'function': {'arguments': '{"keywords": "KABUTO 起動 紫色 点滅 フリーズ"}', 'name': 'search_qdrant'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-11-20', 'system_fingerprint': 'fp_ee1d74bde0'}, id='run--60587ca8-d54e-4c66-8d9c-c5a99ae479a9-0', tool_calls=[{'name': 'search_elasticsearch', 'args': {'keywords': 'KABUTO 起動 紫色 点滅 フリーズ'}, 'id': 'call_LTenxsczZMuCdIAK8IBT9e7r', 'type': 'tool_call'}, {'name': 'search_qdrant', 'args': {'keywords': 'KABUTO 起動 紫色 点滅 フリーズ'}, 'id': 'call_POOYozcQOSjnXEaVkbxQpTz7', 'type': 'tool_call'}])]}} (agent_operator.py:105)
2025-08-05 11:27:41,458 [    INFO] -------------------- (agent_operator.py:104)
2025-08-05 11:27:41,458 [    INFO] Event: {'tools': {'messages': [ToolMessage(content='"[ElasticsearchOutput(file_name=\'docs_kabuto.pdf\', content=\'「⻤灯」の実⾏中は、冷却システムが⼀時的に停⽌し、無⾳となる。これは内部エネルギーの流れを最適化するため\\\\nの仕様であり、異常ではない。ただし、この無⾳状態が 15 分以上継続する場合は、過熱の可能性があるため、アプリ\\\\nケーションの強制終了およびシステムの再起動が推奨される。\\\\n第 3 章  ソフトウェア・オペレーション\\\\n3.1 起動プロトコル\\\\nKABUTO の起動シーケンスは、「シノビ・プロトコル」に基づき実⾏される。このプロトコルの初期化フェーズで、\\\\n内部クロックと接続された外部周辺機器のクロックとの同期に失敗した場合、画⾯全体が紫⾊に点滅し、システムが\'), ElasticsearchOutput(file_name=\'docs_kabuto.pdf\', content=\'2.1 電源・起動システム\\\\nKABUTO の電源ランプは、通常の動作状態を以下のパターンで表⽰する。待機状態では⾚⾊点滅、稼働状態では⻘⾊\\\\n点滅が繰り返される。このパターンにない緑⾊の点滅は、システムが「禅モード」に移⾏していることを⽰す。禅モ\\\\nードでは、パフォーマンスが最⼩限に抑えられ、バックグラウンドでのシステム⾃⼰修復が実⾏される。このモード\\\\nからの強制離脱は、 KABUTO 本体に設置された「⿓⽟」ボタンを 5 秒以上⻑押しすることで可能となる。また、電源\\\\nOFF 後も微細な電流が流れ続ける「夢の海」と呼ばれる待機モードが存在し、これにより次回の起動⾼速化が図られ\'), ElasticsearchOutput(file_name=\'docs_kabuto.pdf\', content=\'フリーズする。この問題は、 KABUTO 本体の再起動前に、すべての周辺機器を 5 秒以上電源から抜くことで、外部ク\\\\nロックのキャッシュがクリアされ、同期が正常に⾏われる可能性が⾼まる。\\\\n3.2 セキュリティ・認証システム\\\\nログイン認証は、パスワード⼊⼒に加え、セキュリティモジュール「百⻤夜⾏」によるログイン試⾏者の「精神波\\\\n形」解析を伴う。パスワードが正しく⼊⼒された場合でも、精神波形が登録されたユーザーと⼀致しないと判断され\')]"', name='search_elasticsearch', id='33ac53bd-22d8-4fe1-9094-bd4869c4771a', tool_call_id='call_LTenxsczZMuCdIAK8IBT9e7r'), ToolMessage(content='"[QdrantOutput(file_name=\'./data/qa_kabuto.csv\', content=\'q: KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。 a: この現象は、「シノビ・プロトコル」が初期化フェーズで、内部クロックと外部クロックの同期に失敗している場合に発生します。KABUTO本体を再起動する前に、必ず接続されているすべての周辺機器を5秒以上、電源から抜いてください。これにより、外部クロックのキャッシュがクリアされ、同期が正常に行われる可能性が高まります。\'), QdrantOutput(file_name=\'./data/qa_kabuto.csv\', content=\'q: KABUTOの電源ランプが、赤と青の交互点滅ではなく、緑色で点滅しています。 a: 通常、電源ランプは赤（待機中）と青（稼働中）の点滅を繰り返します。緑色の点滅は、KABUTOが「禅モード」に入っており、パフォーマンスを最小限に抑え、バックグラウンドでシステム自己修復を行っている状態を示します。この状態では、ユーザー操作はほとんど受け付けられません。このモードから強制的に脱出するには、KABUTO本体の「龍玉」ボタンを5秒以上長押ししてください。\'), QdrantOutput(file_name=\'./data/qa_kabuto.csv\', content=\'q: KABUTOの画面の隅に、時々「狐火」のような光の点が、不規則に現れては消えます。 a: 「狐火」は、KABUTOの演算コア「稲荷」が、膨大な量のタスクを処理しきれずに、一部の計算結果が視覚的なノイズとして現れている現象です。これは、システム負荷が高いことを示しています。この現象が頻繁に発生する場合は、バックグラウンドで動作している不要なアプリケーションを終了させるか、KABUTOを再起動してください。\')]"', name='search_qdrant', id='97878816-f478-4bd5-a78e-0b668311bf05', tool_call_id='call_POOYozcQOSjnXEaVkbxQpTz7')]}} (agent_operator.py:105)
2025-08-05 11:27:45,507 [    INFO] -------------------- (agent_operator.py:104)
2025-08-05 11:27:45,508 [    INFO] Event: {'chat_with_tools': {'messages': [AIMessage(content='KABUTOの起動時に画面全体が紫色に点滅し、システムがフリーズする問題について、以下の情報が得られました。\n\n### 原因\nこの現象は、「シノビ・プロトコル」が初期化フェーズで、内部クロックと外部クロックの同期に失敗している場合に発生します。\n\n### 解決方法\n1. **周辺機器の電源を切る**:\n   - KABUTO本体を再起動する前に、接続されているすべての周辺機器を5秒以上電源から抜いてください。\n   - これにより、外部クロックのキャッシュがクリアされ、同期が正常に行われる可能性が高まります。\n\n2. **強制再起動**:\n   - 必要に応じて、KABUTO本体の「龍玉」ボタンを5秒以上長押しして強制再起動を試みてください。\n\n### 注意点\n- この問題が頻繁に発生する場合は、システムの内部エネルギー流れや冷却システムの状態を確認する必要があります。\n- 再起動後も問題が解決しない場合は、専門の技術者に相談することをお勧めします。\n\nこれらの手順を試してみてください。問題が解決しない場合は、さらに詳細な情報を提供していただければ追加のサポートを行います。', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-11-20', 'system_fingerprint': 'fp_ee1d74bde0'}, id='run--c3c0ecb8-b85f-4072-8420-eeb94f2b62a1-0')]}} (agent_operator.py:105)
```

## Tutorials

<!-- Add docs here -->

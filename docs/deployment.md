## Deployment instructions

### Azure resources

To deploy Azure resources for this project, you can use the provided [Bicep template](https://github.com/ks6088ts-labs/baseline-environment-on-azure-bicep/tree/main/infra/scenarios/template-langgraph). This template sets up the necessary resources for running the LangGraph application.

To quickly deploy the application, [Deploy to Azure button](https://learn.microsoft.com/azure/azure-resource-manager/templates/deploy-to-azure-button) is available. By clicking the following button, you can deploy the resources directly from the Azure portal:

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fks6088ts-labs%2Fbaseline-environment-on-azure-bicep%2Frefs%2Fheads%2Fmain%2Finfra%2Fscenarios%2Ftemplate-langgraph%2Fazuredeploy.json)

### Create index

#### On Azure

```shell
# Cosmos DB
make create-cosmosdb-index

# Azure AI Search
make create-ai-search-index
```

#### On Docker

```shell
rm -rf assets/

# Launch Docker containers
docker compose up -d --wait

# Qdrant
make create-qdrant-index

# Elasticsearch
make create-elasticsearch-index
```

### Agents

#### Create agent graph in png format

```shell
## Draw agent graph
mkdir -p generated
AGENT_NAMES=(
    "chat_with_tools_agent"
    "image_classifier_agent"
    "issue_formatter_agent"
    "kabuto_helpdesk_agent"
    "news_summarizer_agent"
    "task_decomposer_agent"
)
for AGENT_NAME in "${AGENT_NAMES[@]}"; do
    uv run python scripts/agent_operator.py png --name "$AGENT_NAME" --verbose --output "generated/${AGENT_NAME}.png" &
done
wait
```

#### Run agents

```shell
NAME_QUESTION_ARRAY=(
    "chat_with_tools_agent:KABUTOの起動時に、画面全体が紫色に点滅し、システムがフリーズします。KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください"
    "issue_formatter_agent:KABUTOにログインできない！パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。"
    "kabuto_helpdesk_agent:天狗のいたずら という現象について KABUTO のマニュアルから、関連する情報を取得したり過去のシステムのトラブルシュート事例が蓄積されたデータベースから、関連する情報を取得して質問に答えてください"
    "task_decomposer_agent:KABUTOにログインできない！パスワードは合ってるはずなのに…若手社員である山田太郎は、Windows 11 を立ち上げ、日課のように自社の業務システムKABUTOのログイン画面を開きます。しかし、そこには、意味をなさない「虚無」という文字だけがただひっそりと表示されていたのです。これは質問でもあり不具合の報告でもあります。岡本太郎さんに本件調査依頼します。"
)
for NAME_QUESTION in "${NAME_QUESTION_ARRAY[@]}"; do
    IFS=':' read -r AGENT_NAME QUESTION <<< "$NAME_QUESTION"
    echo "Running agent: $AGENT_NAME with question: $QUESTION"
    uv run python scripts/agent_operator.py run --name "$AGENT_NAME" --verbose --question "$QUESTION"
done
```

### Docker Hub

To publish the docker image to Docker Hub, you need to [create access token](https://app.docker.com/settings/personal-access-tokens/create) and set the following secrets in the repository settings.

```shell
gh secret set DOCKERHUB_USERNAME --body $DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN --body $DOCKERHUB_TOKEN
```

### Azure Static Web Apps

```shell
RESOURCE_GROUP_NAME=your-resource-group-name
SWA_NAME=your-static-web-app-name

# Create a static app
az staticwebapp create --name $SWA_NAME --resource-group $RESOURCE_GROUP_NAME

# Retrieve the API key
AZURE_STATIC_WEB_APPS_API_TOKEN=$(az staticwebapp secrets list --name $SWA_NAME --query "properties.apiKey" -o tsv)

# Set the API key as a GitHub secret
gh secret set AZURE_STATIC_WEB_APPS_API_TOKEN --body $AZURE_STATIC_WEB_APPS_API_TOKEN
```

Refer to the following links for more information:

- [Deploying to Azure Static Web App](https://docs.github.com/en/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)
- [Create a static web app: `az staticwebapp create`](https://learn.microsoft.com/en-us/cli/azure/staticwebapp?view=azure-cli-latest#az-staticwebapp-create)

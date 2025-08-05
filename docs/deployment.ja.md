# デプロイメント手順

## Azure リソース

Azure リソースをこのプロジェクトにデプロイするには、提供された [Bicep テンプレート](https://github.com/ks6088ts-labs/baseline-environment-on-azure-bicep/tree/main/infra/scenarios/template-langgraph) を使用できます。このテンプレートは、LangGraph アプリケーションを実行するために必要なリソースをセットアップします。

アプリケーションを迅速にデプロイするために、[Deploy to Azure ボタン](https://learn.microsoft.com/azure/azure-resource-manager/templates/deploy-to-azure-button) が利用可能です。以下のボタンをクリックすることで、Azure ポータルから直接リソースをデプロイできます：

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fks6088ts-labs%2Fbaseline-environment-on-azure-bicep%2Frefs%2Fheads%2Fmain%2Finfra%2Fscenarios%2Ftemplate-langgraph%2Fazuredeploy.json)

## Docker Hub

Docker Hub に Docker イメージを公開するには、[アクセストークンを作成](https://app.docker.com/settings/personal-access-tokens/create)し、リポジトリ設定で以下のシークレットを設定する必要があります。

```shell
gh secret set DOCKERHUB_USERNAME --body $DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN --body $DOCKERHUB_TOKEN
```

## Azure Static Web Apps

```shell
RESOURCE_GROUP_NAME=your-resource-group-name
SWA_NAME=your-static-web-app-name

# 静的アプリを作成する
az staticwebapp create --name $SWA_NAME --resource-group $RESOURCE_GROUP_NAME

# APIキーを取得する
AZURE_STATIC_WEB_APPS_API_TOKEN=$(az staticwebapp secrets list --name $SWA_NAME --query "properties.apiKey" -o tsv)

# APIキーをGitHubシークレットとして設定する
gh secret set AZURE_STATIC_WEB_APPS_API_TOKEN --body $AZURE_STATIC_WEB_APPS_API_TOKEN
```

詳細については、以下のリンクを参照してください：

- [Azure Static Web App へのデプロイ](https://docs.github.com/en/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)
- [静的 Web アプリの作成: `az staticwebapp create`](https://learn.microsoft.com/en-us/cli/azure/staticwebapp?view=azure-cli-latest#az-staticwebapp-create)

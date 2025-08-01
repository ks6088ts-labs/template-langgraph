[![test](https://github.com/ks6088ts/template-python/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/ks6088ts/template-python/actions/workflows/test.yaml?query=branch%3Amain)
[![docker](https://github.com/ks6088ts/template-python/actions/workflows/docker.yaml/badge.svg?branch=main)](https://github.com/ks6088ts/template-python/actions/workflows/docker.yaml?query=branch%3Amain)
[![docker-release](https://github.com/ks6088ts/template-python/actions/workflows/docker-release.yaml/badge.svg)](https://github.com/ks6088ts/template-python/actions/workflows/docker-release.yaml)
[![ghcr-release](https://github.com/ks6088ts/template-python/actions/workflows/ghcr-release.yaml/badge.svg)](https://github.com/ks6088ts/template-python/actions/workflows/ghcr-release.yaml)
[![docs](https://github.com/ks6088ts/template-python/actions/workflows/github-pages.yaml/badge.svg)](https://github.com/ks6088ts/template-python/actions/workflows/github-pages.yaml)

# template-python

This is a template repository for Python

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [GNU Make](https://www.gnu.org/software/make/)

## Development instructions

### Local development

Use Makefile to run the project locally.

```shell
# help
make

# install dependencies for development
make install-deps-dev

# run tests
make test

# run CI tests
make ci-test
```

### Docker development

```shell
# build docker image
make docker-build

# run docker container
make docker-run

# run CI tests in docker container
make ci-test-docker
```

## Deployment instructions

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

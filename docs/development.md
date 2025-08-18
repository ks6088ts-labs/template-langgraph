# Development instructions

## Local development

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

# optional: run LangGraph Studio / FastAPI / Streamlit
make langgraph-studio
make fastapi-dev
make streamlit
```

## Testing

```shell
# Run all tests for AI agents
bash scripts/test_all.sh
```

## Docker development

```shell
# build docker image
make docker-build

# run docker container
make docker-run

# run CI tests in docker container
make ci-test-docker

```

## Documentation

Build and serve docs locally:

```shell
make install-deps-docs
make docs-serve
```

# 開発手順

## ローカル開発

Makefile を使用してプロジェクトをローカルで実行します。

```shell
# ヘルプを表示
make

# 開発用の依存関係をインストール
make install-deps-dev

# テストを実行
make test

# CIテストを実行
make ci-test
```

## テスト

```shell
# AIエージェントのすべてのテストを実行
bash scripts/test_all.sh
```

## Docker 開発

```shell
# Dockerイメージをビルド
make docker-build

# Dockerコンテナを実行
make docker-run

# DockerコンテナでCIテストを実行
make ci-test-docker
```

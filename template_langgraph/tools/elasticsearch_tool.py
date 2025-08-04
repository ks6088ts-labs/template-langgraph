import os
from functools import lru_cache

from elasticsearch import Elasticsearch, helpers
from langchain.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    elasticsearch_url: str = "http://localhost:9200"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_elasticsearch_settings() -> Settings:
    """Get Elasticsearch settings."""
    return Settings()


class ElasticsearchClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_elasticsearch_settings()
        self.client = Elasticsearch(
            settings.elasticsearch_url,
        )
        self.mapping = {
            # ドキュメントのマッピング設定を定義
            "mappings": {
                # ドキュメント内の各フィールドのプロパティを定義
                "properties": {
                    # 'content' フィールドを定義
                    "content": {
                        # 'content' は全文検索用のフィールド
                        "type": "text",  # テキスト検索用のフィールド
                        # 日本語用のカスタムアナライザー 'kuromoji_analyzer' を使用
                        "analyzer": "kuromoji_analyzer",  # 日本語用のアナライザーを指定
                    }
                },
            },
            # インデックスの設定（アナライザーなど）を定義
            "settings": {
                # インデックスの分析設定
                "analysis": {
                    # 使用するアナライザーを定義
                    "analyzer": {
                        # 'kuromoji_analyzer' というカスタムアナライザーを定義
                        "kuromoji_analyzer": {
                            # カスタムアナライザーであることを指定
                            "type": "custom",
                            # ICU正規化（文字の正規化処理）を適用
                            "char_filter": ["icu_normalizer"],
                            # Kuromojiトークナイザー（形態素解析用）を使用
                            "tokenizer": "kuromoji_tokenizer",
                            # トークンに対するフィルタのリストを定義
                            "filter": [
                                # 動詞や形容詞の基本形に変換
                                "kuromoji_baseform",
                                # 品詞に基づいたフィルタリング
                                "kuromoji_part_of_speech",
                                # 日本語のストップワード（不要な単語）を除去
                                "ja_stop",
                                # 数字の正規化を行う
                                "kuromoji_number",
                                # 日本語の語幹（ルート形）を抽出
                                "kuromoji_stemmer",
                            ],
                        }
                    }
                }
            },
        }

    def create_index(self, index_name: str) -> bool:
        """Create an index in Elasticsearch."""
        if not self.client.indices.exists(index=index_name):
            result = self.client.indices.create(index=index_name, body=self.mapping)
            if result:
                return True
        return False

    def delete_index(self, index_name: str) -> bool:
        """Delete an index in Elasticsearch."""
        if self.client.indices.exists(index=index_name):
            result = self.client.indices.delete(index=index_name)
            return result.get("acknowledged", False)
        return False

    def add_documents(
        self,
        index_name: str,
        documents: list[Document],
    ) -> bool:
        """Add documents to an Elasticsearch index."""
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "filename": os.path.basename(doc.metadata.get("source", "unknown")),
                    "content": doc.page_content,
                },
            }
            for doc in documents
        ]
        success, _ = helpers.bulk(self.client, actions)
        return success > 0

    def search(
        self,
        index_name: str,
        query: str,
        max_results: int = 10,
    ) -> list[Document]:
        """Search documents in an Elasticsearch index."""
        search_query = {
            "query": {
                "match": {
                    "content": query,
                }
            },
            "size": max_results,
        }
        response = self.client.search(
            index=index_name,
            body=search_query,
        )
        return [
            Document(
                page_content=hit["_source"]["content"],
                metadata={
                    "source": hit["_source"]["filename"],
                },
            )
            for hit in response["hits"]["hits"]
        ]


class ElasticsearchInput(BaseModel):
    keywords: str = Field(description="Keywords to search")


class ElasticsearchOutput(BaseModel):
    file_name: str = Field(description="The file name")
    content: str = Field(description="The content of the file")


@tool(args_schema=ElasticsearchInput)
def search_elasticsearch(
    keywords: str,
) -> list[ElasticsearchOutput]:
    """
    空想上のシステム「KABUTO」のマニュアルから、関連する情報を取得します。
    """
    wrapper = ElasticsearchClientWrapper()
    results = wrapper.search(
        index_name="docs_kabuto",
        query=keywords,
        max_results=3,
    )
    outputs = []
    for result in results:
        outputs.append(
            ElasticsearchOutput(
                file_name=result.metadata["source"],
                content=result.page_content,
            ),
        )
    return outputs

## FAQ

### Docker Compose で Elasticsearch を起動できない

**現象:**

WSL2 上で `docker compose up elasticsearch` を実行した際に、以下のエラーが発生する

`java.lang.IllegalStateException: failed to obtain node locks, tried [/usr/share/elasticsearch/data]; maybe these locations are not writable or multiple nodes were started on the same data path?`

**原因:**

Elasticsearch がデータディレクトリ（/usr/share/elasticsearch/data）にロックファイルを作成できないことを示しています。

**対処方法:**

ディレクトリの権限を修正するために、以下のコマンドを実行してください。

```shell
sudo chown -R 1000:1000 ./assets/es_data
```

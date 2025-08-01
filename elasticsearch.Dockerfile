FROM docker.elastic.co/elasticsearch/elasticsearch:9.1.0

RUN bin/elasticsearch-plugin install analysis-kuromoji
RUN bin/elasticsearch-plugin install analysis-icu

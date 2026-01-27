#!/usr/bin/env python3

__author__ = "xi"

from agent_types.common import LLMConfig
from agent_types.retrieval import DenseEmbeddingRequest
from elasticsearch import Elasticsearch, helpers
from pydantic import BaseModel

from embedding import EmbeddingAdapter

TEMPLATE_NAME = "objects_template"
INDEX_NAME = "objects_index"


class RawMemory:
    """Raw memory module."""

    class Config(BaseModel):
        embedding_url: str | LLMConfig
        embedding_dims: int
        index_name: str = INDEX_NAME
        template_name: str = TEMPLATE_NAME

    def __init__(self, config: Config):
        self.config = config

        self.embedding_adapter = EmbeddingAdapter(self.config.embedding_url)
        self.es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "tangxx11"))

    def get_index_template(self, name: str = TEMPLATE_NAME):
        for t in self.es.indices.get_index_template()["index_templates"]:
            if t["name"] == name:
                return t
        return None

    def get_index_info(self, name: str = INDEX_NAME):
        for idx in self.es.cat.indices(format="json"):
            if idx["index"] == name:
                return idx
        return None

    def destroy(self):
        if self.es.indices.exists(index=self.config.index_name):
            self.es.indices.delete(index=self.config.index_name)
        if self.es.indices.exists_index_template(name=self.config.template_name):
            self.es.indices.delete_index_template(name=self.config.template_name)

    def initialize(self):
        if self.get_index_template():
            raise RuntimeError("Already initialized.")

        dynamic_templates = [
            {
                "strings_as_keyword": {
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            {
                "longs_as_long": {
                    "match_mapping_type": "long",
                    "mapping": {
                        "type": "long"
                    }
                }
            },
            {
                "floats_as_float": {
                    "match_mapping_type": "double",
                    "mapping": {
                        "type": "float"
                    }
                }
            },
            {
                "booleans": {
                    "match_mapping_type": "boolean",
                    "mapping": {
                        "type": "boolean"
                    }
                }
            },
            {
                "dates": {
                    "match_mapping_type": "date",
                    "mapping": {
                        "type": "date"
                    }
                }
            }
        ]
        properties = {
            "id": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            "vector": {
                "type": "dense_vector",
                "dims": self.config.embedding_dims,
                "index": True,
                "similarity": "cosine"
            }
        }
        template_body = {
            "index_patterns": [self.config.index_name],
            "priority": 100,
            "template": {
                # "settings": {
                #     "number_of_shards": 1,
                #     "number_of_replicas": 0
                # },
                "mappings": {
                    "dynamic": True,
                    "dynamic_templates": dynamic_templates,
                    "properties": properties
                }
            },
            "_meta": {
                "schema_version": 1,
                "description": "Objects index template with keyword-first dynamic mapping"
            }
        }
        self.es.indices.put_index_template(
            name=TEMPLATE_NAME,
            body=template_body
        )

        self.es.indices.create(index=self.config.index_name)

    def insert(self, docs: list[dict] | dict):
        if isinstance(docs, dict):
            docs = [docs]

        text = [doc["text"] for doc in docs]
        response = self.embedding_adapter.embedding(DenseEmbeddingRequest(text=text, normalize=False))
        for doc, vector in zip(docs, response.embedding.to_array().tolist()):
            doc["vector"] = vector

        helpers.bulk(self.es, [
            {"_index": self.config.index_name, "_source": doc}
            for doc in docs
        ])

    def search(self, query: str, k=5, num_candidates=50):
        emb = self.embedding_adapter.embedding(DenseEmbeddingRequest(text=query, normalize=False)).embedding
        query_vector = emb.to_array().tolist()

        resp = self.es.search(
            index=self.config.index_name,
            query={
                "bool": {
                    "must": [
                        {"match": {"text": query}}
                    ]
                }
            },
            knn={
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates
            }
        )
        docs = []
        for hit in resp["hits"]["hits"]:
            docs.append({k: v for k, v in hit["_source"].items() if k != "vector"})
            print(hit["_id"], hit["_score"], docs[-1])


def main():
    data = [
        "Objects index template with keyword-first dynamic mapping",
        "什么是索引模板，什么是动态映射",
        "How to retrieval from a database",
        "今天太冷了",
        "动态索引模板可以用来定义索引的默认类型"
    ]
    service = RawMemory(RawMemory.Config(
        embedding_url=LLMConfig(
            base_url="https://api.zhizengzeng.com/v1",
            api_key="zk-08507a246a7f1fc844185def5ae9ef2b",
            model="text-embedding-3-small"
        ),
        embedding_dims=1536
    ))

    # service.destroy()
    # service.initialize()
    # service.insert([
    #     {"text": item, "type": "knowledge"}
    #     for item in data
    # ])

    service.search("dynamic template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

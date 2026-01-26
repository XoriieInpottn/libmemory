#!/usr/bin/env python3

__author__ = "xi"

from elasticsearch import Elasticsearch

TEMPLATE_NAME = "objects_template"
INDEX_NAME = "objects_index"


class Mem:

    def __init__(self):
        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", "tangxx11")
        )

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
        if self.es.indices.exists(index=INDEX_NAME):
            self.es.indices.delete(index=INDEX_NAME)
        if self.es.indices.exists_index_template(name=TEMPLATE_NAME):
            self.es.indices.delete_index_template(name=TEMPLATE_NAME)

    def init_index(self, vector_dims: int):
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
                "dims": vector_dims,
                "index": True,
                "similarity": "cosine"
            },
            "payload": {"type": "object", "enabled": False}
        }
        template_body = {
            "index_patterns": [INDEX_NAME],
            "priority": 100,
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
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

        self.es.indices.create(index=INDEX_NAME)


def main():
    mem = Mem()
    mem.destroy()
    mem.init_index(1024)
    print(mem.get_index_info())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

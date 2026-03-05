#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
from typing import Any
from uuid import uuid4

import lancedb
from agent_types.common import LLMConfig
from agent_types.retrieval import DenseEmbeddingRequest
from embedding import EmbeddingAdapter
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from pydantic import BaseModel, Field, field_validator

DEFAULT_TABLE_NAME = "knowledge"
DEFAULT_EMBEDDING_DIMS = 1536


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


class KnowledgeDocument(BaseModel):
    id: str = Field(
        default="",
        title="Document ID",
        description="Unique identifier for the document.",
    )
    text: str = Field(
        default="",
        title="Document text",
        description="The content of the document.",
    )
    type: str = Field(
        default="",
        title="Document type",
        description="The business category of the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Document metadata",
        description="Arbitrary metadata associated with the document.",
    )
    created_at: float = Field(
        default=0.0,
        title="Creation timestamp",
        description="UNIX timestamp when the document was created.",
    )

    @field_validator("text", "type")
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text and type cannot be empty")
        return value


class DocumentStore:
    """Knowledge/document storage and semantic retrieval based on LanceDB + embedding service."""

    def __init__(
        self,
        db_path: str,
        embedding_service_url: str | LLMConfig,
        table_name: str = DEFAULT_TABLE_NAME,
        embedding_dims: int = DEFAULT_EMBEDDING_DIMS,
        normalize_embeddings: bool = True,
        ensure_fts_index: bool = True,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dims = embedding_dims
        self.normalize_embeddings = normalize_embeddings

        self.embedding_adapter = EmbeddingAdapter(embedding_service_url)

        self.db = lancedb.connect(self.db_path)
        schema = self._build_schema(self.embedding_dims)
        self.table = self._open_or_create_table(schema)

        if ensure_fts_index:
            self._ensure_fts_index()

    @staticmethod
    def _build_schema(embedding_dims: int):
        class KnowledgeTableDocument(KnowledgeDocument, LanceModel):
            vector: Vector(embedding_dims)

        return KnowledgeTableDocument

    def _open_or_create_table(self, schema):
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            return self.db.create_table(self.table_name, schema=schema)

    def _ensure_fts_index(self) -> None:
        try:
            self.table.create_fts_index("text", replace=False)
        except Exception:
            # Index may already exist; keep the constructor idempotent.
            pass

    def _embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        response = self.embedding_adapter.embedding(
            DenseEmbeddingRequest(text=text, normalize=self.normalize_embeddings)
        )
        return response.embedding.to_array().tolist()

    def _prepare_documents(
        self, documents: list[KnowledgeDocument]
    ) -> list[dict[str, Any]]:
        embeddings = self._embed(
            documents[0].text if len(documents) == 1 else [doc.text for doc in documents]
        )
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]  # type: ignore[arg-type]

        payloads: list[dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            document_id = doc.id or uuid4().hex
            payloads.append(
                {
                    "id": document_id,
                    "text": doc.text,
                    "type": doc.type,
                    "metadata": json.dumps(doc.metadata or {}, ensure_ascii=False),
                    "created_at": doc.created_at or time.time(),
                    "vector": embeddings[idx],
                }
            )
        return payloads

    def _add_documents(
        self, 
        document: KnowledgeDocument | list[KnowledgeDocument], 
        upsert: bool = False
    ) -> str | list[str]:
        """Internal helper to add or upsert documents."""
        is_single = isinstance(document, KnowledgeDocument)
        docs_list = [document] if is_single else document

        if upsert:
            ids_to_delete = [doc.id for doc in docs_list if doc.id]
            if ids_to_delete:
                escaped_ids = [f"'{_escape_sql_literal(id)}'" for id in ids_to_delete]
                self.table.delete(f"id IN ({', '.join(escaped_ids)})")
            
        payloads = self._prepare_documents(docs_list)
        self.table.add(payloads)
        
        document_ids = [p["id"] for p in payloads]
        return document_ids[0] if is_single else document_ids

    def insert_document(
        self, document: KnowledgeDocument | list[KnowledgeDocument]
    ) -> str | list[str]:
        """Purely insert new document(s). 
        
        If a document ID is provided and already exists, this might result in 
        duplicate IDs depending on the underlying storage behavior.
        """
        return self._add_documents(document, upsert=False)

    def upsert_document(
        self, document: KnowledgeDocument | list[KnowledgeDocument]
    ) -> str | list[str]:
        """Insert or update document(s).
        
        If a document ID is provided and already exists, the old document 
        will be deleted before inserting the new one.
        """
        return self._add_documents(document, upsert=True)

    def delete_document(self, document_id: str) -> None:
        """Delete a document by its ID."""
        self.table.delete(f"id = '{_escape_sql_literal(document_id)}'")

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        doc_type: str | None = None,
        where: str | None = None,
    ) -> list[KnowledgeDocument]:
        if vector_weight >= 1.0:
            # 纯向量检索
            query_vector = self._embed(query)
            search_builder = self.table.search(query_vector, query_type="vector")
        elif vector_weight <= 0.0:
            # 纯全文检索
            search_builder = self.table.search(query, query_type="fts")
        else:
            # 混合检索
            query_vector = self._embed(query)
            reranker = LinearCombinationReranker(weight=vector_weight)
            search_builder = (
                self.table.search(query_type="hybrid")
                .vector(query_vector)
                .text(query)
                .rerank(reranker)
            )

        conditions = []
        if doc_type:
            conditions.append(f"type = '{_escape_sql_literal(doc_type)}'")
        if where:
            conditions.append(f"({where})")
        if conditions:
            search_builder = search_builder.where(" AND ".join(conditions))

        rows = search_builder.limit(top_k).to_list()
        results: list[KnowledgeDocument] = []
        for row in rows:
            metadata_raw = row.get("metadata") or "{}"
            try:
                metadata = json.loads(metadata_raw)
            except (TypeError, json.JSONDecodeError):
                metadata = {"_raw_metadata": metadata_raw}

            text_value = row.get("text") or ""
            type_value = row.get("type") or ""
            if not text_value.strip() or not type_value.strip():
                continue

            results.append(
                KnowledgeDocument(
                    id=row.get("id") or "",
                    text=text_value,
                    type=type_value,
                    metadata=metadata,
                    created_at=row.get("created_at") or 0.0,
                )
            )
        return results

    def get_document(self, document_id: str) -> KnowledgeDocument:
        """Retrieve a single document by its ID.
        
        Args:
            document_id: The unique identifier of the document.
            
        Returns:
            The KnowledgeDocument with the specified ID.
            
        Raises:
            ValueError: If no document with the given ID is found.
        """
        results = self.table.search().where(f"id = '{_escape_sql_literal(document_id)}'").limit(1).to_list()
        
        if not results:
            raise ValueError(f"Document with ID '{document_id}' not found")
        
        row = results[0]
        metadata_raw = row.get("metadata") or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except (TypeError, json.JSONDecodeError):
            metadata = {"_raw_metadata": metadata_raw}
        
        return KnowledgeDocument(
            id=row.get("id") or "",
            text=row.get("text") or "",
            type=row.get("type") or "",
            metadata=metadata,
            created_at=row.get("created_at") or 0.0,
        )

    def list_documents(
        self,
        type: str | list[str] | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[KnowledgeDocument]:
        """List documents, optionally filtered by type, with pagination.

        Args:
            type: Optional document type(s) to filter by. Can be a single type string,
                a list of type strings, or None to include all documents.
            skip: Number of documents to skip (offset). Must be >= 0.
            limit: Maximum number of documents to return. ``0`` means no limit
                (return all remaining documents after ``skip``).

        Returns:
            A list of KnowledgeDocument instances matching the filter criteria.
        """
        if skip < 0:
            raise ValueError("skip must be >= 0")
        if limit < 0:
            raise ValueError("limit must be >= 0")

        conditions: list[str] = []

        if type is not None:
            if isinstance(type, str):
                conditions.append(f"type = '{_escape_sql_literal(type)}'")
            elif isinstance(type, list):
                if type:  # Only add condition if list is not empty
                    escaped_types = [f"'{_escape_sql_literal(t)}'" for t in type]
                    conditions.append(f"type IN ({', '.join(escaped_types)})")

        query_builder = self.table.search()
        if conditions:
            query_builder = query_builder.where(" AND ".join(conditions))

        # Apply pagination
        if skip:
            query_builder = query_builder.offset(skip)
        if limit:
            query_builder = query_builder.limit(limit)

        rows = query_builder.to_list()
        documents: list[KnowledgeDocument] = []

        for row in rows:
            metadata_raw = row.get("metadata") or "{}"
            try:
                metadata = json.loads(metadata_raw)
            except (TypeError, json.JSONDecodeError):
                metadata = {"_raw_metadata": metadata_raw}

            text_value = row.get("text") or ""
            type_value = row.get("type") or ""

            documents.append(
                KnowledgeDocument(
                    id=row.get("id") or "",
                    text=text_value,
                    type=type_value,
                    metadata=metadata,
                    created_at=row.get("created_at") or 0.0,
                )
            )

        return documents


def test():
    # 从配置文件加载 LLMConfig（避免在代码中硬编码敏感信息）
    config_path = "./config.json"
    if not os.path.exists(config_path):
        print(f"未找到配置文件 '{config_path}'，请确保该文件存在。")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # LLMConfig 是 Pydantic 模型，可以使用 model_validate 进行验证加载
    try:
        config = LLMConfig.model_validate(cfg)
    except (AttributeError, ValueError):
        # 兼容旧版本 Pydantic 或特定实现
        config = LLMConfig(**cfg)

    # 初始化 DocumentStore
    db_path = "./data/document_db"
    
    # 判断数据库是否已存在，决定是否需要初始化文档
    is_new_db = not os.path.exists(db_path)
    
    store = DocumentStore(
        db_path=db_path,
        embedding_service_url=config,
        table_name="test_knowledge"
    )

    if is_new_db:
        # 准备测试数据（单条插入）
        docs_single = [
            KnowledgeDocument(text="Python 是一种简洁易学的编程语言，广泛用于数据科学", type="tech", metadata={"source": "test"}),
            KnowledgeDocument(text="机器学习是人工智能的核心分支，包含监督学习和无监督学习", type="ai", metadata={"source": "test"}),
        ]

        print("=== 单条插入文档 ===")
        for doc in docs_single:
            doc_id = store.insert_document(doc)
            print(f"已插入: {doc_id} -> {doc.text[:20]}...")

        # 批量插入文档
        docs_batch = [
            KnowledgeDocument(text="向量数据库专门用于存储和检索高维向量数据", type="database", metadata={"source": "batch"}),
            KnowledgeDocument(text="LanceDB 是一个高性能的嵌入式向量数据库，支持混合检索", type="database", metadata={"source": "batch"}),
        ]
        
        print("\n=== 批量插入文档 ===")
        batch_ids = store.insert_document(docs_batch)
        for doc_id, doc in zip(batch_ids, docs_batch):
            print(f"已插入: {doc_id} -> {doc.text[:20]}...")

        # 测试更新 (upsert)
        print("\n=== 测试更新 (upsert) ===")
        if batch_ids:
            target_id = batch_ids[0]
            update_doc = KnowledgeDocument(
                id=target_id,
                text="LanceDB 是一个极其高性能的嵌入式向量数据库，它真的很快！",
                type="database",
                metadata={"source": "upsert_test"}
            )
            upserted_id = store.upsert_document(update_doc)
            print(f"已更新文档 ID: {upserted_id}")
            
            # 验证更新
            check_doc = store.get_document(target_id)
            print(f"验证更新后的内容: {check_doc.text}")
    else:
        print("=== 数据库已存在，跳过插入步骤 ===")

    # 测试检索
    query = "数据库存储向量"
    print(f"\n=== 混合检索测试 (weight=0.7): '{query}' ===")
    results = store.search(query, top_k=3, vector_weight=0.7)
    for i, res in enumerate(results):
        print(f"{i+1}. {res.text} (类型: {res.type}, ID: {res.id})")

    # 测试纯向量检索
    print(f"\n=== 纯向量检索测试 (weight=1.0): '{query}' ===")
    vector_results = store.search(query, top_k=3, vector_weight=1.0)
    for i, res in enumerate(vector_results):
        print(f"{i+1}. {res.text}")

    # 测试带条件的检索
    print(f"\n=== 带过滤条件的混合检索 (type='database'): '{query}' ===")
    filtered_results = store.search(query, top_k=3, doc_type="database")
    for i, res in enumerate(filtered_results):
        print(f"{i+1}. {res.text} (metadata: {res.metadata})")

    # 测试 list_documents（不带过滤，返回全部）
    print("\n=== list_documents 测试（全部文档） ===")
    all_docs = store.list_documents()
    for i, doc in enumerate(all_docs):
        print(f"{i+1}. {doc.text} (类型: {doc.type}, ID: {doc.id})")

    # 测试 list_documents（带分页）
    print("\n=== list_documents 测试（skip=1, limit=2） ===")
    paged_docs = store.list_documents(skip=1, limit=2)
    for i, doc in enumerate(paged_docs):
        print(f"{i+1}. {doc.text} (类型: {doc.type}, ID: {doc.id})")

    # 测试 get_document（根据 ID 精确获取）
    if all_docs:
        print("\n=== get_document 测试 ===")
        first_id = all_docs[0].id
        doc = store.get_document(first_id)
        print(f"根据 ID 获取: {doc.id} -> {doc.text[:20]}... (类型: {doc.type})")


if __name__ == "__main__":
    test()


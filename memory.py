#!/usr/bin/env python3


import os
from typing import List, Optional, Union
from document_store import DocumentStore, KnowledgeDocument
from graph_store import GraphStore
from agent_types.common import LLMConfig


class Memory:

    def __init__(
        self,
        db_root: str = "./data",
        embedding_config: Optional[Union[str, LLMConfig]] = None,
        embedding_dims: int = 1536,
        table_name: str = "knowledge",
    ):
        """
        初始化 Memory 引擎，组合 DocumentStore 和 GraphStore。
        
        Args:
            db_root: 数据库文件存放的根目录。
            embedding_config: Embedding 服务的配置 (URL 字符串或 LLMConfig 对象)。
            embedding_dims: 向量维度，默认 1536。
            table_name: 文档库表名。
        """
        # 确保根目录存在
        os.makedirs(db_root, exist_ok=True)

        # 定义子数据库路径
        doc_db_path = os.path.join(db_root, "document_db")
        graph_db_path = os.path.join(db_root, "graph_db")

        # 初始化文档存储 (LanceDB)
        self.document_store = DocumentStore(
            db_path=doc_db_path,
            embedding_service_url=embedding_config,
            embedding_dims=embedding_dims,
            table_name=table_name
        )

        # 初始化图存储 (Kuzu)
        self.graph_store = GraphStore(db_path=graph_db_path)

    def read(self, query: str, top_k: int = 5, max_hops: int = 3, type: Optional[str] = None) -> List[KnowledgeDocument]:
        """
        根据 query 读取相关记忆。
        
        1. 先从 DocumentStore 中语义检索出 top_k 个种子文档。
        2. 再从 GraphStore 中基于这些种子扩展出相关的文档 IDs。
        3. 最后从 DocumentStore 中获取这些 IDs 对应的完整文档。
        """
        # 1. 语义检索种子文档 (Top K)
        seeds = self.document_store.search(query, top_k=top_k, doc_type=type)
        if not seeds:
            return []

        seed_ids = [doc.id for doc in seeds]
        all_relevant_ids = set(seed_ids)

        # 2. 从图谱中扩展 (基于种子寻找邻居)
        for seed_id in seed_ids:
            # 在图中寻找最大 max_hops 步之内的邻居
            neighbors = self.graph_store.retrieve(node_id=seed_id, max_hops=max_hops, node_type=type)
            for node in neighbors:
                all_relevant_ids.add(node.id)

        # 3. 批量获取文档对象并去重返回
        final_docs = {doc.id: doc for doc in seeds}

        # 补全扩展出的其他文档
        for doc_id in all_relevant_ids:
            if doc_id not in final_docs:
                try:
                    doc = self.document_store.get_document(doc_id)
                    final_docs[doc_id] = doc
                except ValueError:
                    # 如果图中有 ID 但文档库中没有，跳过
                    continue

        return list(final_docs.values())

    def write(self, document: KnowledgeDocument, link_to: Optional[str] = None, relation: Optional[str] = None) -> None:
        """写入文档到 DocumentStore 并同步到 GraphStore。"""
        # 1. 写入文档存储
        doc_id = self.document_store.upsert_document(document)
        if isinstance(doc_id, list):
            doc_id = doc_id[0]

        # 2. 写入图节点
        self.graph_store.insert_node(doc_id, document.type)

        # 3. 如果有指定的关系，创建边
        if link_to and relation:
            self.graph_store.insert_edge(doc_id, link_to, relation)

    def link(self, link_from: str, link_to: str, relation: str) -> None:
        """在图谱中创建两个文档之间的联系。"""
        self.graph_store.insert_edge(link_from, link_to, relation)

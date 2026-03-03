#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


try:
    import kuzu  # type: ignore[import]
except ImportError as e:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "kuzu package is required for GraphStore. "
        "Install it with `pip install kuzu`."
    ) from e


@dataclass
class GraphNode:
    """Simple graph node model used by GraphStore."""

    id: str
    type: str
    distance: int


class GraphStore:
    """Graph storage engine based on Kuzu.

    The graph is modeled with:
    - Node table: Node(id STRING PRIMARY KEY, type STRING)
    - Rel table:  Edge(FROM Node TO Node, relType STRING)

    The semantics of `id` and `type` follow `KnowledgeDocument` in `DocumentStore`.
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Directory path for the Kuzu database.
        """
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create node/edge tables if they do not already exist."""
        # Create node table: stores id and type (same semantics as KnowledgeDocument).
        try:
            self.conn.execute(
                """
                CREATE NODE TABLE Node(
                    id STRING PRIMARY KEY,
                    type STRING
                )
                """
            )
        except Exception:
            # Table may already exist; keep this method idempotent.
            pass

        # Create relationship table: directed edges with a relation type.
        try:
            self.conn.execute(
                """
                CREATE REL TABLE Edge(
                    FROM Node TO Node,
                    relType STRING
                )
                """
            )
        except Exception:
            # Table may already exist; keep this method idempotent.
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_node(self, node_id: str, node_type: str) -> None:
        """Insert a node with deduplication.

        If the node already exists, its type will not be modified.
        """
        if not node_id:
            raise ValueError("node_id cannot be empty")
        if not node_type:
            raise ValueError("node_type cannot be empty")

        # MERGE ensures deduplication based on the primary key `id`.
        # We only set `type` on creation to avoid overwriting existing value.
        self.conn.execute(
            """
            MERGE (n:Node {id: $id})
            ON CREATE SET n.type = $type
            RETURN n.id
            """,
            {"id": node_id, "type": node_type},
        )

    def insert_edge(self, src_id: str, dst_id: str, rel_type: str) -> None:
        """Insert an edge with deduplication.

        The edge is directed: src_id -> dst_id with the given relation type.
        Nodes will be auto-created if they do not exist yet (with empty type).
        """
        if not src_id or not dst_id:
            raise ValueError("src_id and dst_id cannot be empty")
        if not rel_type:
            raise ValueError("rel_type cannot be empty")

        # Ensure endpoint nodes exist; do not overwrite type if already set.
        self.conn.execute(
            """
            MERGE (a:Node {id: $src_id})
            ON CREATE SET a.type = ''
            """,
            {"src_id": src_id},
        )
        self.conn.execute(
            """
            MERGE (b:Node {id: $dst_id})
            ON CREATE SET b.type = ''
            """,
            {"dst_id": dst_id},
        )

        # MERGE relationship pattern to avoid duplicate edges for same
        # (src_id, dst_id, rel_type).
        self.conn.execute(
            """
            MATCH (a:Node {id: $src_id})
            MATCH (b:Node {id: $dst_id})
            MERGE (a)-[e:Edge {relType: $rel_type}]->(b)
            RETURN e.relType
            """,
            {"src_id": src_id, "dst_id": dst_id, "rel_type": rel_type},
        )

    def retrieve(
        self,
        node_id: str,
        max_hops: int,
        node_type: Optional[str] = None,
        rel_type: Optional[str] = None,
    ) -> List[GraphNode]:
        """Retrieve neighbor nodes up to a given hop distance.

        Args:
            node_id: Start node ID.
            max_hops: Maximum hop distance (>= 1).
            node_type: Optional node type filter. If set, only nodes with this
                type will be returned (the start node is always included).
            rel_type: Optional relation type filter. If set, only edges whose
                `relType` equals this value are followed.

        Returns:
            A list of `GraphNode` objects, including the start node with
            distance 0. Each node appears at most once (shortest distance).
        """
        if not node_id:
            raise ValueError("node_id cannot be empty")
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")

        # If max_hops is 0, only return the start node (if it exists).
        if max_hops == 0:
            start = self._get_node(node_id)
            if start is None:
                return []
            if node_type and start.type != node_type:
                # Start node exists but does not match type filter.
                return []
            start.distance = 0
            return [start]

        visited: Dict[str, GraphNode] = {}
        frontier: Set[str] = set()

        # Initialize with the start node.
        start_node = self._get_node(node_id)
        if start_node is None:
            return []
        start_node.distance = 0
        visited[start_node.id] = start_node
        frontier.add(start_node.id)

        # BFS up to max_hops.
        current_distance = 0
        while frontier and current_distance < max_hops:
            next_frontier: Set[str] = set()
            for current_id in frontier:
                neighbors = self._get_neighbors(current_id, rel_type)
                for nid, ntype in neighbors:
                    if nid in visited:
                        continue
                    distance = current_distance + 1
                    node = GraphNode(id=nid, type=ntype, distance=distance)
                    visited[nid] = node
                    next_frontier.add(nid)
            frontier = next_frontier
            current_distance += 1

        # Apply node_type filter (but always keep the start node).
        result: List[GraphNode] = []
        for node in visited.values():
            if node.id == node_id:
                result.append(node)
                continue
            if node_type and node.type != node_type:
                continue
            result.append(node)

        # Sort by distance then id for determinism.
        result.sort(key=lambda n: (n.distance, n.id))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str) -> Optional[GraphNode]:
        """Fetch a single node by ID."""
        result = self.conn.execute(
            """
            MATCH (n:Node {id: $id})
            RETURN n.id AS id, n.type AS type
            """,
            {"id": node_id},
        )
        df = result.get_as_df()
        if df.empty:
            return None
        row = df.iloc[0]
        return GraphNode(id=str(row["id"]), type=str(row["type"]), distance=0)

    def _get_neighbors(
        self, node_id: str, rel_type: Optional[str]
    ) -> List[Tuple[str, str]]:
        """Fetch direct neighbors (1-hop) of a node."""
        if rel_type:
            result = self.conn.execute(
                """
                MATCH (n:Node {id: $id})-[r:Edge {relType: $rel_type}]->(m:Node)
                RETURN DISTINCT m.id AS id, m.type AS type
                """,
                {"id": node_id, "rel_type": rel_type},
            )
        else:
            result = self.conn.execute(
                """
                MATCH (n:Node {id: $id})-[r:Edge]->(m:Node)
                RETURN DISTINCT m.id AS id, m.type AS type
                """,
                {"id": node_id},
            )
        df = result.get_as_df()
        if df.empty:
            return []
        neighbors: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            neighbors.append((str(row["id"]), str(row["type"])))
        return neighbors


def test():
    """Simple manual test for GraphStore.

    This test:
    1. Creates a fresh Kuzu database in `./test_kuzu_graph_db`.
    2. Inserts a small graph with node/edge dedup.
    3. Runs several retrieval queries and prints the results.
    """
    import os
    import shutil

    db_path = "./test_kuzu_graph_db"

    # Start from a clean database directory for repeatable testing.
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    store = GraphStore(db_path=db_path)

    # Insert nodes (with dedup).
    store.insert_node("doc1", "article")
    store.insert_node("doc2", "article")
    store.insert_node("doc3", "note")
    store.insert_node("doc4", "note")
    # Duplicate insertion should not create multiple nodes or overwrite type.
    store.insert_node("doc1", "article")

    # Insert edges (with dedup).
    store.insert_edge("doc1", "doc2", "similar")
    store.insert_edge("doc2", "doc3", "reference")
    store.insert_edge("doc1", "doc3", "similar")
    store.insert_edge("doc3", "doc4", "reference")
    # Duplicate edge.
    store.insert_edge("doc1", "doc2", "similar")

    print("=== 全局检索（最多 2 跳，不限类型、不限关系）===")
    result_all = store.retrieve(node_id="doc1", max_hops=2)
    for node in result_all:
        print(f"id={node.id}, type={node.type}, distance={node.distance}")

    print("\n=== 只跟随 'similar' 关系（最多 2 跳）===")
    result_similar = store.retrieve(node_id="doc1", max_hops=2, rel_type="similar")
    for node in result_similar:
        print(f"id={node.id}, type={node.type}, distance={node.distance}")

    print("\n=== 过滤节点类型为 'note'（最多 3 跳）===")
    result_note = store.retrieve(node_id="doc1", max_hops=3, node_type="note")
    for node in result_note:
        print(f"id={node.id}, type={node.type}, distance={node.distance}")


if __name__ == "__main__":
    test()


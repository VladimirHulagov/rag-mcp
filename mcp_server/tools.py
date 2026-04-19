import logging
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

log = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://qdrant:6333"))


def _collection() -> str:
    return os.environ.get("QDRANT_COLLECTION", "pdf_library")


def search_library(
    query_vector: List[float],
    top_k: int = 5,
    filter_filename: Optional[str] = None,
    filter_file_type: Optional[str] = None,
) -> Dict[str, Any]:
    client = _get_client()
    name = _collection()

    must = []
    if filter_filename:
        must.append(FieldCondition(key="filename", match=MatchValue(value=filter_filename)))
    if filter_file_type:
        must.append(FieldCondition(key="file_type", match=MatchValue(value=filter_file_type)))

    search_filter = Filter(must=must) if must else None

    response = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    hits = []
    for r in response.points:
        p = r.payload or {}
        hits.append({
            "content": p.get("content", ""),
            "score": r.score,
            "filename": p.get("filename", ""),
            "path": p.get("path", ""),
            "page": p.get("page", 0),
            "chunk_index": p.get("chunk_index", 0),
        })

    return {"results": hits, "total": len(hits), "query_vector_dim": len(query_vector)}


def list_indexed_files(
    filter_file_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = _get_client()
    name = _collection()

    must = []
    if filter_file_type:
        must.append(FieldCondition(key="file_type", match=MatchValue(value=filter_file_type)))

    seen = {}
    offset = None
    while True:
        scroll_filter = Filter(must=must) if must else None
        records, offset = client.scroll(
            collection_name=name,
            limit=100,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=["path", "filename", "file_type", "modified_time"],
        )
        for r in records:
            p = r.payload
            path = p.get("path", "")
            if path not in seen:
                seen[path] = {
                    "path": path,
                    "filename": p.get("filename", ""),
                    "file_type": p.get("file_type", ""),
                    "modified_time": p.get("modified_time", 0),
                    "chunk_count": 0,
                }
            seen[path]["chunk_count"] += 1
        if offset is None:
            break
    return list(seen.values())


def get_file_status(path: str) -> Dict[str, Any]:
    client = _get_client()
    name = _collection()
    records, _ = client.scroll(
        collection_name=name,
        scroll_filter=Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=path))]
        ),
        limit=100,
        with_payload=["filename", "modified_time", "file_type", "page"],
    )
    if not records:
        return {"indexed": False, "path": path}
    p = records[0].payload
    return {
        "indexed": True,
        "path": path,
        "filename": p.get("filename", ""),
        "chunk_count": len(records),
        "modified_time": p.get("modified_time", 0),
        "file_type": p.get("file_type", ""),
    }


def _outline_collection() -> str:
    return os.environ.get("OUTLINE_QDRANT_COLLECTION", "outline_docs")


def search_outline(
    query_vector: List[float],
    top_k: int = 5,
    filter_collection_id: Optional[str] = None,
) -> Dict[str, Any]:
    client = _get_client()
    name = _outline_collection()

    if not client.collection_exists(name):
        return {"results": [], "total": 0, "query_vector_dim": len(query_vector)}

    must = []
    if filter_collection_id:
        must.append(FieldCondition(key="collection_id", match=MatchValue(value=filter_collection_id)))

    search_filter = Filter(must=must) if must else None

    response = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    hits = []
    for r in response.points:
        p = r.payload or {}
        hits.append({
            "title": p.get("title", ""),
            "outline_id": p.get("outline_id", ""),
            "content": p.get("content", ""),
            "score": r.score,
            "chunk_index": p.get("chunk_index", 0),
        })

    return {"results": hits, "total": len(hits), "query_vector_dim": len(query_vector)}


def list_outline_documents(
    filter_collection_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = _get_client()
    name = _outline_collection()

    if not client.collection_exists(name):
        return []

    must = []
    if filter_collection_id:
        must.append(FieldCondition(key="collection_id", match=MatchValue(value=filter_collection_id)))

    seen: Dict[str, Dict[str, Any]] = {}
    offset = None
    while True:
        scroll_filter = Filter(must=must) if must else None
        records, offset = client.scroll(
            collection_name=name,
            limit=100,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=["outline_id", "title", "collection_id", "updated_at"],
        )
        for r in records:
            p = r.payload or {}
            oid = p.get("outline_id", "")
            if oid not in seen:
                seen[oid] = {
                    "outline_id": oid,
                    "title": p.get("title", ""),
                    "collection_id": p.get("collection_id", ""),
                    "updated_at": p.get("updated_at", 0),
                    "chunk_count": 0,
                }
            seen[oid]["chunk_count"] += 1
        if offset is None:
            break
    return list(seen.values())

import logging
import os

from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server import Server

from mcp import types
from .tools import search_library, list_indexed_files, get_file_status, search_outline, list_outline_documents

log = logging.getLogger(__name__)

server = Server("rag-mcp")
sse = SseServerTransport("/messages/")

_BEARER_PREFIX = "Bearer "


def _check_auth(scope):
    token = os.environ.get("MCP_BEARER_TOKEN", "")
    if not token:
        return True
    headers = {}
    for key, value in scope.get("headers", []):
        headers[key.decode()] = value.decode()
    auth_header = headers.get("authorization", "")
    if not auth_header.startswith(_BEARER_PREFIX):
        return False
    return auth_header[len(_BEARER_PREFIX):] == token


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="search_library",
            description="Search the indexed document library using semantic similarity. Returns matching text chunks with source file, page, and relevance score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                    "filter_filename": {"type": "string", "description": "Filter by filename (optional)", "default": None},
                    "filter_file_type": {"type": "string", "description": "Filter by file type: pdf, md, txt, csv (optional)", "default": None},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_indexed_files",
            description="List all indexed files in the library with their metadata and chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_file_type": {"type": "string", "description": "Filter by file type (optional)", "default": None},
                },
            },
        ),
        types.Tool(
            name="get_file_status",
            description="Get indexing status of a specific file: whether it is indexed, how many chunks, last modification time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path in Nextcloud (e.g. /Documents/report.pdf)"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="search_outline",
            description="Search Outline knowledge base documents using semantic similarity. Returns matching Markdown text chunks with document title and relevance score. Use this instead of mcp_outline_* for reading documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                    "filter_collection_id": {"type": "string", "description": "Filter by Outline collection ID (optional)", "default": None},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_outline_documents",
            description="List all indexed Outline documents with their titles, collection IDs, and chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_collection_id": {"type": "string", "description": "Filter by Outline collection ID (optional)", "default": None},
                },
            },
        ),
    ]


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    model = SentenceTransformer(model_name)
    return model


_embedder = None


def _embed_query(text: str):
    global _embedder
    if _embedder is None:
        _embedder = _get_embedder()
    vector = _embedder.encode([text], normalize_embeddings=True)
    return vector[0].tolist()


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_library":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        filter_filename = arguments.get("filter_filename")
        filter_file_type = arguments.get("filter_file_type")
        query_vector = _embed_query(query)
        result = search_library(query_vector, top_k, filter_filename, filter_file_type)
        result["query"] = query
        return [types.TextContent(type="text", text=str(result))]
    elif name == "list_indexed_files":
        filter_file_type = arguments.get("filter_file_type")
        result = list_indexed_files(filter_file_type)
        return [types.TextContent(type="text", text=str(result))]
    elif name == "get_file_status":
        path = arguments["path"]
        result = get_file_status(path)
        return [types.TextContent(type="text", text=str(result))]
    elif name == "search_outline":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        filter_collection_id = arguments.get("filter_collection_id")
        query_vector = _embed_query(query)
        result = search_outline(query_vector, top_k, filter_collection_id)
        result["query"] = query
        return [types.TextContent(type="text", text=str(result))]
    elif name == "list_outline_documents":
        filter_collection_id = arguments.get("filter_collection_id")
        result = list_outline_documents(filter_collection_id)
        return [types.TextContent(type="text", text=str(result))]
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def _send_unauthorized(scope, receive, send):
    body = b'{"error":"unauthorized"}'
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [[b"content-type", b"application/json"], [b"content-length", str(len(body)).encode()]],
    })
    await send({"type": "http.response.body", "body": body})


_http_transport = StreamableHTTPServerTransport(
    mcp_session_id=None,
    is_json_response_enabled=True,
)

import asyncio

async def _run_http_server():
    async with _http_transport.connect() as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )

_http_task = None

async def _ensure_http_server():
    global _http_task
    if _http_task is None:
        _http_task = asyncio.ensure_future(_run_http_server())
        await asyncio.sleep(0.1)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    path = scope.get("path", "")

    if path == "/sse":
        if not _check_auth(scope):
            await _send_unauthorized(scope, receive, send)
            return
        async with sse.connect_sse(scope, receive, send) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
    elif path.startswith("/messages/"):
        if not _check_auth(scope):
            await _send_unauthorized(scope, receive, send)
            return
        await sse.handle_post_message(scope, receive, send)
    elif path == "/mcp":
        if not _check_auth(scope):
            await _send_unauthorized(scope, receive, send)
            return
        await _ensure_http_server()
        await _http_transport.handle_request(scope, receive, send)
    else:
        body = b"Not Found"
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [[b"content-type", b"text/plain"], [b"content-length", str(len(body)).encode()]],
        })
        await send({"type": "http.response.body", "body": body})

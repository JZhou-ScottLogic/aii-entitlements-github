"""
GitHub Claude API
-----------------
pip install fastapi uvicorn "anthropic[mcp]" mcp
GITHUB_TOKEN=ghp_... uvicorn app:app --reload
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Conversation history — grows with every request
conversation_history: list[dict] = []


# ---------------------------------------------------------------------------
# MCP server config
# ---------------------------------------------------------------------------
# Binary variant — download github-mcp-server.exe from:
#   https://github.com/github/github-mcp-server/releases
# and place it alongside this file (or anywhere on PATH).
#
# Docker alternative (requires Docker Desktop):
#   command="docker",
#   args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
#         "ghcr.io/github/github-mcp-server"],
GITHUB_MCP = StdioServerParameters(
    command="github-mcp-server.exe",
    args=["stdio", "--toolsets", "all"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ["GITHUB_TOKEN"]},
)

SYSTEM = """
    You are a GitHub assistant. Use the provided tools to answer questions and take actions on GitHub. When the user's request is ambiguous, prefer read-only tools over write ones.
"""

# ---------------------------------------------------------------------------
# Shared MCP session — started once, reused across all requests
# ---------------------------------------------------------------------------
_mcp_session: ClientSession | None = None
_mcp_tools: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_session, _mcp_tools, _stdio_ctx, _session_ctx

    _stdio_ctx = stdio_client(GITHUB_MCP)
    read, write = await _stdio_ctx.__aenter__()

    _session_ctx = ClientSession(read, write)
    _mcp_session = await _session_ctx.__aenter__()
    await _mcp_session.initialize()

    tools_result = await _mcp_session.list_tools()
    _mcp_tools = [async_mcp_tool(t, _mcp_session) for t in tools_result.tools]
    print(f"Loaded {len(_mcp_tools)} GitHub MCP tools")

    yield

    await _session_ctx.__aexit__(None, None, None)
    await _stdio_ctx.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="GitHub Claude API", lifespan=lifespan)
client = anthropic.AsyncAnthropic()


class Query(BaseModel):
    text: str


class QueryResponse(BaseModel):
    response: str


@app.get("/tools")
async def list_tools():
    return {"tools": [t.name for t in _mcp_tools],
            "total": _mcp_tools.__len__()}


@app.post("/query", response_model=QueryResponse)
async def query(req: Query):
    if _mcp_session is None:
        raise HTTPException(status_code=503, detail="MCP session not ready")

    conversation_history.append({"role": "user", "content": req.text})

    runner = client.beta.messages.tool_runner(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=conversation_history,
        tools=_mcp_tools,
    )

    final_text = ""
    async for message in runner:
        for block in message.content:
            if block.type == "text":
                final_text = block.text

    conversation_history.append({"role": "assistant", "content": final_text})

    return QueryResponse(response=final_text)

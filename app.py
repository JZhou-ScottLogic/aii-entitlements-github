"""
GitHub Claude API
-----------------
pip install fastapi uvicorn "anthropic[mcp]" mcp httpx
GITHUB_TOKEN=ghp_... uvicorn app:app --reload

MCP servers
  • github-mcp-server.exe          - official GitHub MCP server (read/write tools)
  • mcp/github_privileges_mcp_server.py - custom server for org membership,
                                          team membership, and repo collaborator
                                          management (tools missing from the exe)
"""
import os
import sys
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
# ── Server 1: official GitHub MCP binary ───────────────────────────────────
# Download github-mcp-server.exe from:
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

# ── Server 2: custom privileges server ──────────────────────────────────────
# Handles org membership, team membership, and repo collaborator management —
# capabilities absent from the official github-mcp-server binary.
PRIVILEGES_MCP = StdioServerParameters(
    command=sys.executable,
    args=["mcp/github_privileges_mcp_server.py"],
    env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
)

SYSTEM = f"""
    You are a GitHub assistant. Use the provided tools to answer questions
    and take actions on GitHub. You have access to two sets of tools:

    1. General GitHub tools (repos, issues, PRs, code search, etc.) from the
       official github-mcp-server.
    2. Privilege management tools from github-privileges-mcp-server for managing
       organisation membership, team membership, and repository collaborators.

    When the user's request is ambiguous, prefer read-only tools over write ones.
    For any action that grants access (adding members, collaborators, or team
    members), confirm the target org/repo/team and permission level before acting.
"""

# ---------------------------------------------------------------------------
# Shared MCP sessions — started once, reused across all requests
# ---------------------------------------------------------------------------
_mcp_session: ClientSession | None = None          # github-mcp-server.exe
_privileges_session: ClientSession | None = None   # github-privileges-mcp-server
_mcp_tools: list = []                              # merged tool list from both servers


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_session, _privileges_session, _mcp_tools
    global _stdio_ctx, _session_ctx, _priv_stdio_ctx, _priv_session_ctx

    # ── Server 1: github-mcp-server.exe ─────────────────────────────────────
    _stdio_ctx = stdio_client(GITHUB_MCP)
    read, write = await _stdio_ctx.__aenter__()
    _session_ctx = ClientSession(read, write)
    _mcp_session = await _session_ctx.__aenter__()
    await _mcp_session.initialize()

    tools_result = await _mcp_session.list_tools()
    github_tools = [async_mcp_tool(t, _mcp_session) for t in tools_result.tools]
    print(f"[github-mcp-server]            loaded {len(github_tools)} tools")

    # ── Server 2: github-privileges-mcp-server ───────────────────────────────
    _priv_stdio_ctx = stdio_client(PRIVILEGES_MCP)
    priv_read, priv_write = await _priv_stdio_ctx.__aenter__()
    _priv_session_ctx = ClientSession(priv_read, priv_write)
    _privileges_session = await _priv_session_ctx.__aenter__()
    await _privileges_session.initialize()

    priv_tools_result = await _privileges_session.list_tools()
    privileges_tools = [
        async_mcp_tool(t, _privileges_session) for t in priv_tools_result.tools
    ]
    print(f"[github-privileges-mcp-server] loaded {len(privileges_tools)} tools")

    # ── Merge both tool lists ────────────────────────────────────────────────
    _mcp_tools = github_tools + privileges_tools
    print(f"Total tools available to Claude: {len(_mcp_tools)}")

    yield

    # ── Shutdown: close both sessions cleanly ────────────────────────────────
    await _priv_session_ctx.__aexit__(None, None, None)
    await _priv_stdio_ctx.__aexit__(None, None, None)
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
    """Return all tools available to Claude, grouped by source server."""
    privileges_tool_names = {
        "add_org_member", "remove_org_member", "list_org_members",
        "list_org_teams", "create_org_team",
        "add_team_member", "remove_team_member", "list_team_members",
        "add_repo_collaborator", "remove_repo_collaborator", "list_repo_collaborators",
    }
    github_tools = [t.name for t in _mcp_tools if t.name not in privileges_tool_names]
    privilege_tools = [t.name for t in _mcp_tools if t.name in privileges_tool_names]
    return {
        "total": len(_mcp_tools),
        "github_mcp_server": github_tools,
        "github_privileges_mcp_server": privilege_tools,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: Query):
    if _mcp_session is None or _privileges_session is None:
        raise HTTPException(status_code=503, detail="MCP sessions not ready")

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

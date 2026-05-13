"""
github-privileges-mcp-server
-----------------------------
A stdio MCP server that exposes GitHub organisation, team, and repository
collaborator management tools that are absent from the official
github-mcp-server binary.

All tools call the GitHub REST API directly via httpx using the token
supplied through the GITHUB_TOKEN environment variable.

Run standalone (stdio transport):
    GITHUB_TOKEN=ghp_... python mcp/github_privileges_mcp_server.py

Tools exposed
─────────────
Organisation membership
  • add_org_member          – invite / set membership role for a user
  • remove_org_member       – remove a user from an organisation
  • list_org_members        – list all members of an organisation

Team management
  • list_org_teams          – list all teams in an organisation
  • create_org_team         – create a new team in an organisation
  • add_team_member         – add a user to an org team
  • remove_team_member      – remove a user from an org team
  • list_team_members       – list all members of an org team

Repository collaborators
  • add_repo_collaborator   – add a user as a collaborator on a repository
  • remove_repo_collaborator– remove a collaborator from a repository
  • list_repo_collaborators – list all collaborators on a repository
"""

import os
import sys
import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "github-privileges-mcp-server",
    instructions=(
        "Use these tools to manage GitHub organisation membership, "
        "org team membership, and repository collaborators. "
        "Always prefer the least-privileged permission level unless the user "
        "explicitly requests a higher one."
    ),
)

GITHUB_API = "https://api.github.com"


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _raise_for_status(response: httpx.Response) -> None:
    """Raise a descriptive RuntimeError on non-2xx responses."""
    if response.is_error:
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"GitHub API error {response.status_code} — {detail}"
        )


# ---------------------------------------------------------------------------
# Organisation membership tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_org_member(org: str, username: str, role: str = "member") -> dict:
    """
    Invite a user to a GitHub organisation or update their membership role.

    The user will receive an invitation email if they are not already a member.
    Once they accept, their state changes from 'pending' to 'active'.

    Args:
        org:      Organisation login name (e.g. 'my-company').
        username: GitHub username to invite.
        role:     Membership role — 'member' (default) or 'admin'.

    Returns:
        The GitHub membership object containing state, role, and user details.
    """
    if role not in ("member", "admin"):
        raise ValueError(f"Invalid role '{role}'. Must be 'member' or 'admin'.")

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{GITHUB_API}/orgs/{org}/memberships/{username}",
            headers=_headers(),
            json={"role": role},
        )
        _raise_for_status(response)
        return response.json()


@mcp.tool()
async def remove_org_member(org: str, username: str) -> dict:
    """
    Remove a user from a GitHub organisation.

    This will remove the user from all teams within the organisation and
    revoke their access to all organisation repositories.

    Args:
        org:      Organisation login name.
        username: GitHub username to remove.

    Returns:
        A status dict with the HTTP status code confirming removal.
    """
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{GITHUB_API}/orgs/{org}/members/{username}",
            headers=_headers(),
        )
        _raise_for_status(response)
        return {"status": response.status_code, "message": f"Removed {username} from {org}."}


@mcp.tool()
async def list_org_members(
    org: str,
    role: str = "all",
    filter: str = "all",
) -> list[dict]:
    """
    List all members of a GitHub organisation.

    Args:
        org:    Organisation login name.
        role:   Filter by role — 'all' (default), 'admin', or 'member'.
        filter: Filter by 2FA status — 'all' (default) or '2fa_disabled'.

    Returns:
        A list of user objects (login, id, avatar_url, type, site_admin).
    """
    if role not in ("all", "admin", "member"):
        raise ValueError(f"Invalid role filter '{role}'. Must be 'all', 'admin', or 'member'.")

    params: dict = {"per_page": 100, "role": role, "filter": filter}
    members: list[dict] = []

    async with httpx.AsyncClient() as client:
        url = f"{GITHUB_API}/orgs/{org}/members"
        while url:
            response = await client.get(url, headers=_headers(), params=params)
            _raise_for_status(response)
            members.extend(response.json())
            # Follow GitHub pagination via Link header
            url = _next_page(response)
            params = {}  # params are encoded in the next URL

    return members


# ---------------------------------------------------------------------------
# Team management tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_org_teams(org: str) -> list[dict]:
    """
    List all teams in a GitHub organisation.

    Args:
        org: Organisation login name.

    Returns:
        A list of team objects containing id, name, slug, description,
        privacy, and permission fields.
    """
    teams: list[dict] = []
    params: dict = {"per_page": 100}

    async with httpx.AsyncClient() as client:
        url = f"{GITHUB_API}/orgs/{org}/teams"
        while url:
            response = await client.get(url, headers=_headers(), params=params)
            _raise_for_status(response)
            teams.extend(response.json())
            url = _next_page(response)
            params = {}

    return teams


@mcp.tool()
async def create_org_team(
    org: str,
    name: str,
    description: str = "",
    privacy: str = "secret",
    permission: str = "pull",
    parent_team_id: int | None = None,
) -> dict:
    """
    Create a new team in a GitHub organisation.

    Args:
        org:            Organisation login name.
        name:           Team name (must be unique within the org).
        description:    Optional description for the team.
        privacy:        'secret' (default, visible only to members) or 'closed'
                        (visible to all org members).
        permission:     Default repo permission — 'pull' (default), 'triage',
                        'push', 'maintain', or 'admin'.
        parent_team_id: Optional ID of a parent team to nest this team under.

    Returns:
        The newly created team object.
    """
    if privacy not in ("secret", "closed"):
        raise ValueError(f"Invalid privacy '{privacy}'. Must be 'secret' or 'closed'.")
    if permission not in ("pull", "triage", "push", "maintain", "admin"):
        raise ValueError(f"Invalid permission '{permission}'.")

    body: dict = {
        "name": name,
        "description": description,
        "privacy": privacy,
        "permission": permission,
    }
    if parent_team_id is not None:
        body["parent_team_id"] = parent_team_id

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{GITHUB_API}/orgs/{org}/teams",
            headers=_headers(),
            json=body,
        )
        _raise_for_status(response)
        return response.json()


@mcp.tool()
async def add_team_member(
    org: str,
    team_slug: str,
    username: str,
    role: str = "member",
) -> dict:
    """
    Add a user to a team within a GitHub organisation.

    The user must already be a member of the organisation (or have a pending
    invitation) before they can be added to a team.

    Args:
        org:       Organisation login name.
        team_slug: Team slug (URL-friendly version of the team name).
        username:  GitHub username to add.
        role:      Team role — 'member' (default) or 'maintainer'.

    Returns:
        The team membership object containing state and role.
    """
    if role not in ("member", "maintainer"):
        raise ValueError(f"Invalid role '{role}'. Must be 'member' or 'maintainer'.")

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{GITHUB_API}/orgs/{org}/teams/{team_slug}/memberships/{username}",
            headers=_headers(),
            json={"role": role},
        )
        _raise_for_status(response)
        return response.json()


@mcp.tool()
async def remove_team_member(org: str, team_slug: str, username: str) -> dict:
    """
    Remove a user from a team in a GitHub organisation.

    Removing a user from a team does not remove them from the organisation.

    Args:
        org:       Organisation login name.
        team_slug: Team slug.
        username:  GitHub username to remove.

    Returns:
        A status dict confirming removal.
    """
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{GITHUB_API}/orgs/{org}/teams/{team_slug}/memberships/{username}",
            headers=_headers(),
        )
        _raise_for_status(response)
        return {
            "status": response.status_code,
            "message": f"Removed {username} from team {team_slug} in {org}.",
        }


@mcp.tool()
async def list_team_members(
    org: str,
    team_slug: str,
    role: str = "all",
) -> list[dict]:
    """
    List all members of a team in a GitHub organisation.

    Args:
        org:       Organisation login name.
        team_slug: Team slug.
        role:      Filter by team role — 'all' (default), 'member',
                   or 'maintainer'.

    Returns:
        A list of user objects belonging to the team.
    """
    if role not in ("all", "member", "maintainer"):
        raise ValueError(
            f"Invalid role filter '{role}'. Must be 'all', 'member', or 'maintainer'."
        )

    members: list[dict] = []
    params: dict = {"per_page": 100, "role": role}

    async with httpx.AsyncClient() as client:
        url = f"{GITHUB_API}/orgs/{org}/teams/{team_slug}/members"
        while url:
            response = await client.get(url, headers=_headers(), params=params)
            _raise_for_status(response)
            members.extend(response.json())
            url = _next_page(response)
            params = {}

    return members


# ---------------------------------------------------------------------------
# Repository collaborator tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_repo_collaborator(
    owner: str,
    repo: str,
    username: str,
    permission: str = "push",
) -> dict:
    """
    Add a user as a collaborator on a GitHub repository.

    For organisation-owned repositories the user receives an invitation.
    For personal repositories they are added directly.

    Args:
        owner:      Repository owner (user or org login).
        repo:       Repository name.
        username:   GitHub username to add.
        permission: Access level — 'pull', 'triage', 'push' (default),
                    'maintain', or 'admin'.

    Returns:
        The invitation object (for org repos) or a status dict (for personal
        repos where the addition is immediate).
    """
    valid_permissions = ("pull", "triage", "push", "maintain", "admin")
    if permission not in valid_permissions:
        raise ValueError(
            f"Invalid permission '{permission}'. "
            f"Must be one of: {', '.join(valid_permissions)}."
        )

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{GITHUB_API}/repos/{owner}/{repo}/collaborators/{username}",
            headers=_headers(),
            json={"permission": permission},
        )
        _raise_for_status(response)
        # 201 = invitation created, 204 = already a collaborator / added directly
        if response.status_code == 204 or not response.content:
            return {
                "status": response.status_code,
                "message": f"{username} added as collaborator on {owner}/{repo}.",
            }
        return response.json()


@mcp.tool()
async def remove_repo_collaborator(owner: str, repo: str, username: str) -> dict:
    """
    Remove a user as a collaborator from a GitHub repository.

    Args:
        owner:    Repository owner (user or org login).
        repo:     Repository name.
        username: GitHub username to remove.

    Returns:
        A status dict confirming removal.
    """
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{GITHUB_API}/repos/{owner}/{repo}/collaborators/{username}",
            headers=_headers(),
        )
        _raise_for_status(response)
        return {
            "status": response.status_code,
            "message": f"Removed {username} as collaborator from {owner}/{repo}.",
        }


@mcp.tool()
async def list_repo_collaborators(
    owner: str,
    repo: str,
    affiliation: str = "all",
    permission: str | None = None,
) -> list[dict]:
    """
    List all collaborators on a GitHub repository.

    Args:
        owner:       Repository owner (user or org login).
        repo:        Repository name.
        affiliation: Filter — 'outside' (external collaborators only),
                     'direct' (explicitly granted access), or 'all' (default).
        permission:  Optional filter by permission level — 'pull', 'triage',
                     'push', 'maintain', or 'admin'.

    Returns:
        A list of user objects with a nested 'permissions' field.
    """
    if affiliation not in ("outside", "direct", "all"):
        raise ValueError(
            f"Invalid affiliation '{affiliation}'. Must be 'outside', 'direct', or 'all'."
        )

    collaborators: list[dict] = []
    params: dict = {"per_page": 100, "affiliation": affiliation}
    if permission:
        params["permission"] = permission

    async with httpx.AsyncClient() as client:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/collaborators"
        while url:
            response = await client.get(url, headers=_headers(), params=params)
            _raise_for_status(response)
            collaborators.extend(response.json())
            url = _next_page(response)
            params = {}

    return collaborators


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------


def _next_page(response: httpx.Response) -> str | None:
    """Parse the GitHub Link header and return the 'next' URL, or None."""
    link_header = response.headers.get("link", "")
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            url_part = part.split(";")[0].strip()
            return url_part.strip("<>")
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # stdio transport — consumed by app.py via StdioServerParameters

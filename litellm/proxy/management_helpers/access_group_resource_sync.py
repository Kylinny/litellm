"""
Keep `litellm_accessgrouptable.access_agent_ids` and `access_mcp_server_ids`
pointing at agents and MCP servers that still exist.

Access groups store agent and MCP server ids, so deleting an asset without
touching the arrays strands every group on an id nothing serves any more.
"""

from typing import Final

from litellm.proxy.management_helpers.access_group_model_sync import raw_executor
from litellm.proxy.management_helpers.access_group_team_sync import invalidate_access_group_caches
from litellm.types.llms.base import LiteLLMBaseModel


class _TouchedGroupRow(LiteLLMBaseModel):
    access_group_id: str


_REMOVE_AGENT_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_agent_ids" = array_remove("access_agent_ids", $1) '
    'WHERE $1 = ANY("access_agent_ids") '
    'RETURNING "access_group_id"'
)

_REMOVE_MCP_SERVER_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_mcp_server_ids" = array_remove("access_mcp_server_ids", $1) '
    'WHERE $1 = ANY("access_mcp_server_ids") '
    'RETURNING "access_group_id"'
)


async def _rewrite_groups(prisma_client: object, sql: str, resource_id: str) -> None:
    executor: Final = raw_executor(prisma_client)
    touched_rows: Final = await executor.query_raw(sql, resource_id)
    await invalidate_access_group_caches(
        tuple(_TouchedGroupRow.model_validate(row).access_group_id for row in touched_rows)
    )


async def sync_access_groups_for_deleted_agent(prisma_client: object, *, agent_id: str) -> None:
    await _rewrite_groups(prisma_client, _REMOVE_AGENT_ID_SQL, agent_id)


async def sync_access_groups_for_deleted_mcp_server(prisma_client: object, *, server_id: str) -> None:
    await _rewrite_groups(prisma_client, _REMOVE_MCP_SERVER_ID_SQL, server_id)

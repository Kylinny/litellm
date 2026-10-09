from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_resource_sync import (
    sync_access_groups_for_deleted_agent,
    sync_access_groups_for_deleted_mcp_server,
)

_INVALIDATE = "litellm.proxy.management_helpers.access_group_resource_sync.invalidate_access_group_caches"


def _routed_prisma_client(touched_group_ids):
    async def query_raw(sql, *params):
        return [{"access_group_id": group_id} for group_id in touched_group_ids]

    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    writer_inner.query_raw = AsyncMock(side_effect=query_raw)
    reader_inner.query_raw = AsyncMock(side_effect=query_raw)
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    return SimpleNamespace(db=routing), writer_inner, reader_inner


def _access_group_updates(writer_inner):
    return [
        call
        for call in writer_inner.query_raw.await_args_list
        if call.args[0].startswith('UPDATE "LiteLLM_AccessGroupTable"')
    ]


@pytest.mark.asyncio
async def test_delete_agent_removes_agent_id_from_groups_listing_it():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(("ag-1",))

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_agent(prisma_client, agent_id="agent-123")

    (update_call,) = _access_group_updates(writer_inner)
    assert "access_agent_ids" in update_call.args[0]
    assert "access_mcp_server_ids" not in update_call.args[0]
    assert "array_remove" in update_call.args[0]
    assert update_call.args[1:] == ("agent-123",)
    invalidate.assert_awaited_once_with(("ag-1",))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_agent_writes_nothing_when_no_group_lists_the_agent():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(())

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_agent(prisma_client, agent_id="agent-123")

    (update_call,) = _access_group_updates(writer_inner)
    assert update_call.args[1:] == ("agent-123",)
    invalidate.assert_awaited_once_with(())
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_mcp_server_removes_server_id_from_groups_listing_it():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(("ag-1", "ag-2"))

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_mcp_server(prisma_client, server_id="mcp-1")

    (update_call,) = _access_group_updates(writer_inner)
    assert "access_mcp_server_ids" in update_call.args[0]
    assert "access_agent_ids" not in update_call.args[0]
    assert "array_remove" in update_call.args[0]
    assert update_call.args[1:] == ("mcp-1",)
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_mcp_server_writes_nothing_when_no_group_lists_the_server():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(())

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_mcp_server(prisma_client, server_id="mcp-1")

    (update_call,) = _access_group_updates(writer_inner)
    assert update_call.args[1:] == ("mcp-1",)
    invalidate.assert_awaited_once_with(())
    reader_inner.query_raw.assert_not_awaited()

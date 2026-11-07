from __future__ import annotations

import asyncio
from types import SimpleNamespace

from xdc.coordinator.tunnel import TunnelHub, TunnelUnavailable


class FakeWebSocket:
    def __init__(self):
        self.client_state = SimpleNamespace(name="CONNECTED")
        self.sent = []
        self.closed = False

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code=1000, reason=""):
        self.closed = True


def test_replaced_tunnel_does_not_fail_new_connection():
    async def scenario():
        hub = TunnelHub()
        first = await hub.attach("node-1", FakeWebSocket())
        old_future = asyncio.get_running_loop().create_future()
        hub.pending["old"] = ("node-1", old_future)

        second = await hub.attach("node-1", FakeWebSocket())
        new_future = asyncio.get_running_loop().create_future()
        hub.pending["new"] = ("node-1", new_future)

        await hub.detach(first)
        assert not new_future.done(), "old reader failed the replacement's RPC"
        assert hub.connections["node-1"] is second

        await hub.detach(second)
        assert new_future.done()
        assert isinstance(new_future.exception(), TunnelUnavailable)

    asyncio.run(scenario())

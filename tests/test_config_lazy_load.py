import asyncio

from app.core.config import Config


def test_ensure_loaded_only_calls_load_once():
    cfg = Config()
    called = 0

    async def _fake_load():
        nonlocal called
        called += 1
        cfg._loaded = True

    cfg.load = _fake_load

    async def _run():
        await cfg.ensure_loaded()
        await cfg.ensure_loaded()

    asyncio.run(_run())

    assert called == 1

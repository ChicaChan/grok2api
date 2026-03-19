import asyncio

from app.services.token.manager import TokenManager
from app.services.token.models import BASIC__DEFAULT_QUOTA, SUPER_DEFAULT_QUOTA, TokenInfo
from app.services.token.pool import TokenPool


def test_add_super_token_uses_super_default_quota():
    manager = TokenManager()

    async def _fake_save():
        return None

    manager._save = _fake_save

    asyncio.run(manager.add("token-super", "ssoSuper"))

    token = manager.pools["ssoSuper"].get("token-super")
    assert token is not None
    assert token.quota == SUPER_DEFAULT_QUOTA


def test_reset_all_uses_pool_specific_default_quota():
    manager = TokenManager()
    basic_pool = TokenPool("ssoBasic")
    basic_pool.add(TokenInfo(token="basic-token", quota=1))
    super_pool = TokenPool("ssoSuper")
    super_pool.add(TokenInfo(token="super-token", quota=1))
    manager.pools = {
        "ssoBasic": basic_pool,
        "ssoSuper": super_pool,
    }

    async def _fake_save():
        return None

    manager._save = _fake_save

    asyncio.run(manager.reset_all())

    assert basic_pool.get("basic-token").quota == BASIC__DEFAULT_QUOTA
    assert super_pool.get("super-token").quota == SUPER_DEFAULT_QUOTA

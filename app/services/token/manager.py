"""Token manager service."""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

from app.core.config import get_config
from app.core.logger import logger
from app.core.storage import get_storage
from app.services.token.models import (
    BASIC__DEFAULT_QUOTA,
    SUPER_DEFAULT_QUOTA,
    EffortType,
    FAIL_THRESHOLD,
    TokenInfo,
    TokenPoolStats,
    TokenStatus,
)
from app.services.token.pool import TokenPool


REFRESH_BATCH_SIZE = 10
REFRESH_CONCURRENCY = 5
SUPER_POOL_NAME = "ssoSuper"
BASIC_POOL_NAME = "ssoBasic"


def _default_quota_for_pool(pool_name: str) -> int:
    if pool_name == SUPER_POOL_NAME:
        return SUPER_DEFAULT_QUOTA
    return BASIC__DEFAULT_QUOTA


def _refresh_interval_hours_for_pool(pool_name: str) -> float:
    if pool_name == SUPER_POOL_NAME:
        return get_config("token.super_refresh_interval_hours", 2)
    return get_config("token.refresh_interval_hours", 8)


class TokenManager:
    """Manage token pools and quota synchronization."""

    _instance: Optional["TokenManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pools: Dict[str, TokenPool] = {}
        self.initialized = False
        self._save_lock = asyncio.Lock()
        self._dirty = False
        self._save_task: Optional[asyncio.Task] = None
        self._save_delay = 0.5
        self._last_reload_at = 0.0

    @classmethod
    async def get_instance(cls) -> "TokenManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance._load()
        return cls._instance

    async def _load(self):
        if self.initialized:
            return

        try:
            storage = get_storage()
            data = await storage.load_tokens()

            if not data:
                from app.core.storage import LocalStorage

                local_storage = LocalStorage()
                local_data = await local_storage.load_tokens()
                if local_data:
                    data = local_data
                    await storage.save_tokens(local_data)
                    logger.info(
                        f"Initialized remote token storage ({storage.__class__.__name__}) with local tokens."
                    )
                else:
                    data = {}

            self.pools = {}
            for pool_name, tokens in data.items():
                pool = TokenPool(pool_name)
                for token_data in tokens:
                    quota_missing = not (
                        isinstance(token_data, dict) and "quota" in token_data
                    )
                    try:
                        if isinstance(token_data, dict):
                            raw_token = token_data.get("token")
                            if isinstance(raw_token, str) and raw_token.startswith("sso="):
                                token_data["token"] = raw_token[4:]
                        token_info = TokenInfo(**token_data)
                        if quota_missing and pool_name == SUPER_POOL_NAME:
                            token_info.quota = SUPER_DEFAULT_QUOTA
                        pool.add(token_info)
                    except Exception as exc:
                        logger.warning(f"Failed to load token in pool '{pool_name}': {exc}")
                        continue
                pool._rebuild_index()
                self.pools[pool_name] = pool

            self.initialized = True
            self._last_reload_at = time.monotonic()
            total = sum(pool.count() for pool in self.pools.values())
            logger.info(
                f"TokenManager initialized: {len(self.pools)} pools with {total} tokens"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize TokenManager: {exc}")
            self.pools = {}
            self.initialized = True

    async def reload(self):
        async with self.__class__._lock:
            self.initialized = False
            await self._load()

    async def reload_if_stale(self):
        interval = get_config("token.reload_interval_sec", 30)
        try:
            interval = float(interval)
        except Exception:
            interval = 30.0
        if interval <= 0:
            return
        if time.monotonic() - self._last_reload_at < interval:
            return
        await self.reload()

    async def _save(self):
        async with self._save_lock:
            try:
                data = {
                    pool_name: [info.model_dump() for info in pool.list()]
                    for pool_name, pool in self.pools.items()
                }
                storage = get_storage()
                async with storage.acquire_lock("tokens_save", timeout=10):
                    await storage.save_tokens(data)
            except Exception as exc:
                logger.error(f"Failed to save tokens: {exc}")

    def _schedule_save(self):
        delay_ms = get_config("token.save_delay_ms", 500)
        try:
            delay_ms = float(delay_ms)
        except Exception:
            delay_ms = 500
        self._save_delay = max(0.0, delay_ms / 1000.0)
        self._dirty = True
        if self._save_delay == 0:
            if self._save_task and not self._save_task.done():
                return
            self._save_task = asyncio.create_task(self._save())
            return
        if self._save_task and not self._save_task.done():
            return
        self._save_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        try:
            while True:
                await asyncio.sleep(self._save_delay)
                if not self._dirty:
                    break
                self._dirty = False
                await self._save()
        finally:
            self._save_task = None
            if self._dirty:
                self._schedule_save()

    @staticmethod
    def _extract_cookie_value(cookie_str: str, name: str) -> str | None:
        needle = f"{name}="
        if needle not in cookie_str:
            return None
        for part in cookie_str.split(";"):
            part = part.strip()
            if part.startswith(needle):
                value = part[len(needle) :].strip()
                return value or None
        return None

    @classmethod
    def _normalize_input_token(cls, token_str: str) -> str:
        raw = str(token_str or "").strip()
        if not raw:
            return ""
        if ";" in raw:
            return (cls._extract_cookie_value(raw, "sso") or "").strip()
        if raw.startswith("sso="):
            return raw[4:].strip()
        return raw

    def _find_token_info(self, token_str: str) -> tuple[Optional[TokenInfo], str]:
        raw_token = self._normalize_input_token(token_str)
        if not raw_token:
            return None, ""
        for pool in self.pools.values():
            token = pool.get(raw_token)
            if token:
                return token, raw_token
        return None, raw_token

    def get_token(self, pool_name: str = BASIC_POOL_NAME) -> Optional[str]:
        pool = self.pools.get(pool_name)
        if not pool:
            logger.warning(f"Pool '{pool_name}' not found")
            return None

        token_info = pool.select()
        if not token_info:
            logger.warning(f"No available token in pool '{pool_name}'")
            return None

        token = token_info.token
        return token[4:] if token.startswith("sso=") else token

    def get_token_for_model(self, model_id: str) -> Optional[str]:
        from app.services.grok.model import ModelService

        bucket = "heavy" if ModelService.is_heavy_bucket_model(model_id) else "normal"
        for pool_name in ModelService.pool_candidates_for_model(model_id):
            pool = self.pools.get(pool_name)
            if not pool:
                continue
            token_info = pool.select(bucket=bucket)
            if not token_info:
                continue
            token = token_info.token
            return token[4:] if token.startswith("sso=") else token

        logger.warning(f"No available token for model '{model_id}'")
        return None

    async def consume(
        self,
        token_str: str,
        effort: EffortType = EffortType.LOW,
        bucket: str = "normal",
    ) -> bool:
        raw_token = token_str.replace("sso=", "")

        for pool in self.pools.values():
            token = pool.get(raw_token)
            if token:
                consumed = (
                    token.consume_heavy(effort)
                    if bucket == "heavy"
                    else token.consume(effort)
                )
                logger.debug(
                    f"Token {raw_token[:10]}...: consumed {consumed} quota "
                    f"(bucket={bucket}), use_count={token.use_count}"
                )
                self._schedule_save()
                return True

        logger.warning(f"Token {raw_token[:10]}...: not found for consumption")
        return False

    async def sync_usage(
        self,
        token_str: str,
        model_id: str,
        fallback_effort: EffortType = EffortType.LOW,
        consume_on_fail: bool = True,
        is_usage: bool = True,
    ) -> bool:
        raw_token = token_str.replace("sso=", "")

        target_token: Optional[TokenInfo] = None
        for pool in self.pools.values():
            target_token = pool.get(raw_token)
            if target_token:
                break

        if not target_token:
            logger.warning(f"Token {raw_token[:10]}...: not found for sync")
            return False

        from app.services.grok.model import ModelService
        from app.services.grok.usage import UsageService

        bucket = "heavy" if ModelService.is_heavy_bucket_model(model_id) else "normal"
        rate_limit_model = ModelService.rate_limit_model_for(model_id)

        try:
            usage_service = UsageService()
            result = await usage_service.get(token_str, model_name=rate_limit_model)

            if result and "remainingTokens" in result:
                try:
                    new_quota = int(result["remainingTokens"])
                except Exception:
                    new_quota = 0

                if bucket == "heavy":
                    old_quota = target_token.heavy_quota
                    target_token.update_heavy_quota(new_quota)
                else:
                    old_quota = target_token.quota
                    target_token.update_quota(new_quota)

                target_token.record_success(is_usage=is_usage)

                consumed = max(0, old_quota - new_quota) if old_quota >= 0 else 0
                logger.info(
                    f"Token {raw_token[:10]}...: synced quota "
                    f"(bucket={bucket}, model={rate_limit_model}) "
                    f"{old_quota} -> {new_quota} "
                    f"(consumed: {consumed}, use_count: {target_token.use_count})"
                )
                self._schedule_save()
                return True
        except Exception as exc:
            logger.warning(
                f"Token {raw_token[:10]}...: API sync failed, fallback to local ({exc})"
            )

        if consume_on_fail:
            logger.debug(f"Token {raw_token[:10]}...: using local consumption")
            return await self.consume(token_str, fallback_effort, bucket=bucket)

        logger.debug(
            f"Token {raw_token[:10]}...: sync failed, skipping local consumption"
        )
        return False

    async def record_fail(
        self, token_str: str, status_code: int = 401, reason: str = ""
    ) -> bool:
        raw_token = token_str.replace("sso=", "")

        for pool in self.pools.values():
            token = pool.get(raw_token)
            if token:
                if status_code == 401:
                    token.record_fail(status_code, reason)
                    logger.warning(
                        f"Token {raw_token[:10]}...: recorded 401 failure "
                        f"({token.fail_count}/{FAIL_THRESHOLD}) - {reason}"
                    )
                else:
                    logger.info(
                        f"Token {raw_token[:10]}...: non-401 error "
                        f"({status_code}) - {reason} (not counted)"
                    )
                self._schedule_save()
                return True

        logger.warning(f"Token {raw_token[:10]}...: not found for failure record")
        return False

    async def add(self, token: str, pool_name: str = BASIC_POOL_NAME) -> bool:
        if pool_name not in self.pools:
            self.pools[pool_name] = TokenPool(pool_name)
            logger.info(f"Pool '{pool_name}': created")

        pool = self.pools[pool_name]
        token = token[4:] if token.startswith("sso=") else token
        if pool.get(token):
            logger.warning(f"Pool '{pool_name}': token already exists")
            return False

        pool.add(TokenInfo(token=token, quota=_default_quota_for_pool(pool_name)))
        await self._save()
        logger.info(f"Pool '{pool_name}': token added")
        return True

    async def mark_asset_clear(self, token: str) -> bool:
        info, _ = self._find_token_info(token)
        if info:
            info.last_asset_clear_at = int(datetime.now().timestamp() * 1000)
            self._schedule_save()
            return True
        return False

    async def set_token_invalid(
        self, token_str: str, reason: str = "", save: bool = True
    ) -> bool:
        token, raw_token = self._find_token_info(token_str)
        if not token:
            logger.warning(f"Token {raw_token[:10]}...: not found for invalidation")
            return False

        token.status = TokenStatus.EXPIRED
        token.fail_count = max(token.fail_count, FAIL_THRESHOLD)
        token.last_fail_at = int(datetime.now().timestamp() * 1000)
        if reason:
            token.last_fail_reason = str(reason)[:500]

        if save:
            await self._save()
        return True

    async def mark_token_account_settings_success(
        self, token_str: str, save: bool = True
    ) -> bool:
        token, raw_token = self._find_token_info(token_str)
        if not token:
            logger.warning(
                f"Token {raw_token[:10]}...: not found for account-settings success"
            )
            return False

        token.fail_count = 0
        token.last_fail_at = None
        token.last_fail_reason = None
        token.last_sync_at = int(datetime.now().timestamp() * 1000)
        token.status = TokenStatus.COOLING if token.quota == 0 else TokenStatus.ACTIVE

        if save:
            await self._save()
        return True

    async def commit(self):
        await self._save()

    async def remove(self, token: str) -> bool:
        for pool_name, pool in self.pools.items():
            if pool.remove(token):
                await self._save()
                logger.info(f"Pool '{pool_name}': token removed")
                return True

        logger.warning("Token not found for removal")
        return False

    async def reset_all(self):
        count = 0
        for pool_name, pool in self.pools.items():
            default_quota = _default_quota_for_pool(pool_name)
            for token in pool:
                token.reset(default_quota)
                count += 1

        await self._save()
        logger.info(f"Reset all: {count} tokens updated")

    async def reset_token(self, token_str: str) -> bool:
        raw_token = token_str.replace("sso=", "")

        for pool in self.pools.values():
            token = pool.get(raw_token)
            if token:
                default_quota = _default_quota_for_pool(pool.name)
                token.reset(default_quota)
                await self._save()
                logger.info(f"Token {raw_token[:10]}...: reset completed")
                return True

        logger.warning(f"Token {raw_token[:10]}...: not found for reset")
        return False

    def get_stats(self) -> Dict[str, dict]:
        stats = {}
        for name, pool in self.pools.items():
            pool_stats: TokenPoolStats = pool.get_stats()
            stats[name] = pool_stats.model_dump()
        return stats

    def get_pool_tokens(self, pool_name: str = BASIC_POOL_NAME) -> List[TokenInfo]:
        pool = self.pools.get(pool_name)
        if not pool:
            return []
        return pool.list()

    async def refresh_cooling_tokens(self) -> Dict[str, int]:
        from app.services.grok.usage import UsageService

        to_refresh: List[TokenInfo] = []
        for pool_name, pool in self.pools.items():
            interval_hours = _refresh_interval_hours_for_pool(pool_name)
            for token in pool:
                if token.need_refresh(interval_hours):
                    to_refresh.append(token)

        if not to_refresh:
            logger.debug("Refresh check: no tokens need refresh")
            return {"checked": 0, "refreshed": 0, "recovered": 0, "expired": 0}

        logger.info(f"Refresh check: found {len(to_refresh)} cooling tokens to refresh")

        semaphore = asyncio.Semaphore(REFRESH_CONCURRENCY)
        usage_service = UsageService()
        refreshed = 0
        recovered = 0
        expired = 0

        async def _refresh_one(token_info: TokenInfo) -> dict:
            async with semaphore:
                token_str = token_info.token
                if token_str.startswith("sso="):
                    token_str = token_str[4:]

                for retry in range(3):
                    try:
                        result = await usage_service.get(token_str)

                        if result and "remainingTokens" in result:
                            new_quota = result["remainingTokens"]
                            old_quota = token_info.quota
                            old_status = token_info.status

                            token_info.update_quota(new_quota)
                            token_info.mark_synced()

                            logger.info(
                                f"Token {token_info.token[:10]}...: refreshed "
                                f"{old_quota} -> {new_quota}, status: "
                                f"{old_status} -> {token_info.status}"
                            )

                            return {
                                "recovered": new_quota > 0 and old_quota == 0,
                                "expired": False,
                            }

                        token_info.mark_synced()
                        return {"recovered": False, "expired": False}
                    except Exception as exc:
                        error_str = str(exc)

                        if "401" in error_str or "Unauthorized" in error_str:
                            if retry < 2:
                                logger.warning(
                                    f"Token {token_info.token[:10]}...: 401 error, "
                                    f"retry {retry + 1}/2..."
                                )
                                await asyncio.sleep(0.5)
                                continue

                            logger.error(
                                f"Token {token_info.token[:10]}...: 401 after 2 retries, "
                                "marking as expired"
                            )
                            token_info.status = TokenStatus.EXPIRED
                            token_info.mark_synced()
                            return {"recovered": False, "expired": True}

                        logger.warning(
                            f"Token {token_info.token[:10]}...: refresh failed ({exc})"
                        )
                        token_info.mark_synced()
                        return {"recovered": False, "expired": False}

                token_info.mark_synced()
                return {"recovered": False, "expired": False}

        for index in range(0, len(to_refresh), REFRESH_BATCH_SIZE):
            batch = to_refresh[index : index + REFRESH_BATCH_SIZE]
            results = await asyncio.gather(*[_refresh_one(token) for token in batch])
            refreshed += len(batch)
            recovered += sum(result["recovered"] for result in results)
            expired += sum(result["expired"] for result in results)

            if index + REFRESH_BATCH_SIZE < len(to_refresh):
                await asyncio.sleep(1)

        await self._save()

        logger.info(
            f"Refresh completed: checked={len(to_refresh)}, refreshed={refreshed}, "
            f"recovered={recovered}, expired={expired}"
        )

        return {
            "checked": len(to_refresh),
            "refreshed": refreshed,
            "recovered": recovered,
            "expired": expired,
        }


async def get_token_manager() -> TokenManager:
    return await TokenManager.get_instance()


__all__ = ["TokenManager", "get_token_manager"]

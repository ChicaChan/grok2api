"""
Token data models.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


BASIC__DEFAULT_QUOTA = 80
SUPER_DEFAULT_QUOTA = 140
DEFAULT_QUOTA = BASIC__DEFAULT_QUOTA

FAIL_THRESHOLD = 5


class TokenStatus(str, Enum):
    """Token lifecycle status."""

    ACTIVE = "active"
    DISABLED = "disabled"
    EXPIRED = "expired"
    COOLING = "cooling"


class EffortType(str, Enum):
    """Local quota cost level."""

    LOW = "low"
    HIGH = "high"


EFFORT_COST = {
    EffortType.LOW: 1,
    EffortType.HIGH: 4,
}


class TokenInfo(BaseModel):
    """Runtime token state."""

    token: str
    status: TokenStatus = TokenStatus.ACTIVE
    quota: int = BASIC__DEFAULT_QUOTA
    heavy_quota: int = -1

    created_at: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )
    last_used_at: Optional[int] = None
    use_count: int = 0

    fail_count: int = 0
    last_fail_at: Optional[int] = None
    last_fail_reason: Optional[str] = None

    last_sync_at: Optional[int] = None

    tags: List[str] = Field(default_factory=list)
    note: str = ""
    last_asset_clear_at: Optional[int] = None

    def is_available(self) -> bool:
        return self.status == TokenStatus.ACTIVE and self.quota > 0

    def consume(self, effort: EffortType = EffortType.LOW) -> int:
        cost = EFFORT_COST[effort]
        actual_cost = min(cost, self.quota)

        self.last_used_at = int(datetime.now().timestamp() * 1000)
        self.use_count += 1
        self.quota = max(0, self.quota - cost)

        self.fail_count = 0
        self.last_fail_reason = None

        if self.quota == 0:
            self.status = TokenStatus.COOLING
        elif self.status in [TokenStatus.COOLING, TokenStatus.EXPIRED]:
            self.status = TokenStatus.ACTIVE

        return actual_cost

    def update_quota(self, new_quota: int):
        self.quota = max(0, new_quota)

        if self.quota == 0:
            self.status = TokenStatus.COOLING
        elif self.quota > 0 and self.status in [
            TokenStatus.COOLING,
            TokenStatus.EXPIRED,
        ]:
            self.status = TokenStatus.ACTIVE

    def update_heavy_quota(self, new_quota: int):
        try:
            value = int(new_quota)
        except Exception:
            value = 0
        self.heavy_quota = max(0, value)

    def consume_heavy(self, effort: EffortType = EffortType.LOW) -> int:
        cost = EFFORT_COST[effort]

        self.last_used_at = int(datetime.now().timestamp() * 1000)
        self.use_count += 1
        self.fail_count = 0
        self.last_fail_reason = None

        if self.heavy_quota < 0:
            return 0

        actual_cost = min(cost, self.heavy_quota)
        self.heavy_quota = max(0, self.heavy_quota - actual_cost)
        return actual_cost

    def reset(self, default_quota: Optional[int] = None):
        quota = BASIC__DEFAULT_QUOTA if default_quota is None else default_quota
        self.quota = max(0, int(quota))
        self.heavy_quota = -1
        self.status = TokenStatus.ACTIVE
        self.fail_count = 0
        self.last_fail_reason = None

    def record_fail(self, status_code: int = 401, reason: str = ""):
        if status_code != 401:
            return

        self.fail_count += 1
        self.last_fail_at = int(datetime.now().timestamp() * 1000)
        self.last_fail_reason = reason

        if self.fail_count >= FAIL_THRESHOLD:
            self.status = TokenStatus.EXPIRED

    def record_success(self, is_usage: bool = True):
        self.fail_count = 0
        self.last_fail_at = None
        self.last_fail_reason = None

        if is_usage:
            self.use_count += 1
            self.last_used_at = int(datetime.now().timestamp() * 1000)

        if self.quota == 0:
            self.status = TokenStatus.COOLING
        else:
            self.status = TokenStatus.ACTIVE

    def need_refresh(self, interval_hours: int = 8) -> bool:
        if self.status != TokenStatus.COOLING:
            return False

        if self.last_sync_at is None:
            return True

        now = int(datetime.now().timestamp() * 1000)
        interval_ms = interval_hours * 3600 * 1000
        return (now - self.last_sync_at) >= interval_ms

    def mark_synced(self):
        self.last_sync_at = int(datetime.now().timestamp() * 1000)


class TokenPoolStats(BaseModel):
    """Aggregate pool stats."""

    total: int = 0
    active: int = 0
    disabled: int = 0
    expired: int = 0
    cooling: int = 0
    total_quota: int = 0
    avg_quota: float = 0.0


__all__ = [
    "TokenStatus",
    "TokenInfo",
    "TokenPoolStats",
    "EffortType",
    "EFFORT_COST",
    "BASIC__DEFAULT_QUOTA",
    "SUPER_DEFAULT_QUOTA",
    "DEFAULT_QUOTA",
    "FAIL_THRESHOLD",
]

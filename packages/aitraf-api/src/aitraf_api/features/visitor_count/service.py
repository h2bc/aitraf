"""Redis-backed visitor-count operations."""

from __future__ import annotations

from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

VISITOR_COUNT_KEY = "aitraf:visitor-count:homepage"


class VisitorCountUnavailableError(RuntimeError):
    """Raised when Redis cannot complete a visitor-count operation."""


class InvalidVisitorCountError(RuntimeError):
    """Raised when visitor-count state violates the required representation."""


class VisitorCounter(Protocol):
    async def validate(self) -> None: ...

    async def increment(self) -> int: ...

    async def aclose(self) -> None: ...


def validate_increment_result(value: int) -> int:
    if type(value) is not int or value < 0:
        raise InvalidVisitorCountError("Redis returned an invalid visitor count")
    return value


def validate_stored_count(value: str) -> None:
    if not value.isascii() or not value.isdecimal():
        raise InvalidVisitorCountError("Redis contains an invalid visitor count")


class RedisVisitorCounter:
    def __init__(self, client: Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, redis_url: str) -> RedisVisitorCounter:
        return cls(Redis.from_url(redis_url, decode_responses=True))

    async def validate(self) -> None:
        try:
            await self._client.ping()
            stored_type = await self._client.type(VISITOR_COUNT_KEY)
            if stored_type not in {"none", "string"}:
                raise InvalidVisitorCountError(
                    "Redis contains an invalid visitor count type"
                )
            value = await self._client.get(VISITOR_COUNT_KEY)
        except RedisError as error:
            raise VisitorCountUnavailableError(
                "Redis visitor count is unavailable"
            ) from error
        if value is not None:
            validate_stored_count(value)

    async def increment(self) -> int:
        try:
            value = await self._client.incr(VISITOR_COUNT_KEY)
        except ResponseError as error:
            raise InvalidVisitorCountError(
                "Redis contains an invalid visitor count"
            ) from error
        except RedisError as error:
            raise VisitorCountUnavailableError(
                "Redis visitor count is unavailable"
            ) from error
        return validate_increment_result(value)

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = [
    "InvalidVisitorCountError",
    "RedisVisitorCounter",
    "VISITOR_COUNT_KEY",
    "VisitorCounter",
    "VisitorCountUnavailableError",
    "validate_increment_result",
    "validate_stored_count",
]

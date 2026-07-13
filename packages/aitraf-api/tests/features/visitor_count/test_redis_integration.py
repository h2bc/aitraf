from __future__ import annotations

import asyncio
import os

import pytest
from redis.asyncio import Redis

from aitraf_api.features.visitor_count.service import (
    InvalidVisitorCountError,
    RedisVisitorCounter,
    VISITOR_COUNT_KEY,
    VisitorCountUnavailableError,
)

pytestmark = pytest.mark.redis_integration


def _redis_url() -> str:
    return os.environ["AITRAF_REDIS_URL"]


def test_real_redis_creates_key_and_preserves_concurrent_increments() -> None:
    async def run() -> None:
        raw = Redis.from_url(_redis_url(), decode_responses=True)
        counter = RedisVisitorCounter(raw)
        try:
            await raw.delete(VISITOR_COUNT_KEY)
            assert await counter.increment() == 1
            results = await asyncio.gather(*(counter.increment() for _ in range(100)))
            assert len(set(results)) == 100
            assert await raw.get(VISITOR_COUNT_KEY) == "101"
        finally:
            await raw.delete(VISITOR_COUNT_KEY)
            await counter.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_real_redis_rejects_invalid_stored_values(value: str) -> None:
    async def run() -> None:
        raw = Redis.from_url(_redis_url(), decode_responses=True)
        counter = RedisVisitorCounter(raw)
        try:
            await raw.set(VISITOR_COUNT_KEY, value)
            with pytest.raises(InvalidVisitorCountError):
                await counter.validate()
        finally:
            await raw.delete(VISITOR_COUNT_KEY)
            await counter.aclose()

    asyncio.run(run())


def test_real_redis_rejects_alternate_stored_type() -> None:
    async def run() -> None:
        raw = Redis.from_url(_redis_url(), decode_responses=True)
        counter = RedisVisitorCounter(raw)
        try:
            await raw.rpush(VISITOR_COUNT_KEY, "1")
            with pytest.raises(InvalidVisitorCountError):
                await counter.validate()
        finally:
            await raw.delete(VISITOR_COUNT_KEY)
            await counter.aclose()

    asyncio.run(run())


def test_unavailable_redis_fails_explicitly() -> None:
    async def run() -> None:
        counter = RedisVisitorCounter.from_url(
            "redis://127.0.0.1:1/0?socket_connect_timeout=0.05"
        )
        try:
            with pytest.raises(VisitorCountUnavailableError):
                await counter.validate()
        finally:
            await counter.aclose()

    asyncio.run(run())

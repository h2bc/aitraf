from __future__ import annotations

import asyncio

import pytest

from aitraf_api.features.visitor_count.service import (
    InvalidVisitorCountError,
    RedisVisitorCounter,
    VISITOR_COUNT_KEY,
    validate_increment_result,
    validate_stored_count,
)


class FakeRedis:
    def __init__(self) -> None:
        self.value: int | None = None

    async def incr(self, key: str) -> int:
        assert key == VISITOR_COUNT_KEY
        self.value = 1 if self.value is None else self.value + 1
        return self.value


def test_first_and_sequential_increments_return_new_count() -> None:
    async def run() -> tuple[int, int]:
        counter = RedisVisitorCounter(FakeRedis())  # type: ignore[arg-type]
        return await counter.increment(), await counter.increment()

    assert asyncio.run(run()) == (1, 2)


@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_increment_result_rejects_invalid_values(value: object) -> None:
    with pytest.raises(InvalidVisitorCountError):
        validate_increment_result(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["-1", "1.0", "value", " 1", "+1"])
def test_stored_count_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidVisitorCountError):
        validate_stored_count(value)


def test_stored_count_accepts_non_negative_integer() -> None:
    validate_stored_count("0")
    validate_stored_count("42")

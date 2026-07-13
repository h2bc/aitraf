"""Public visitor-count route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from aitraf_api.features.visitor_count.schemas import VisitorCountResponse
from aitraf_api.features.visitor_count.service import (
    InvalidVisitorCountError,
    VisitorCounter,
    VisitorCountUnavailableError,
)

router = APIRouter(tags=["visitor-count"])


async def require_empty_request(request: Request) -> None:
    if request.query_params or await request.body():
        raise HTTPException(
            status_code=422, detail="Request body and query are not supported"
        )


@router.post("/visitor-count", response_model=VisitorCountResponse)
async def increment_visitor_count(request: Request) -> VisitorCountResponse:
    await require_empty_request(request)
    counter: VisitorCounter = request.app.state.visitor_counter
    try:
        count = await counter.increment()
    except VisitorCountUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except InvalidVisitorCountError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return VisitorCountResponse(count=count)


__all__ = ["router"]

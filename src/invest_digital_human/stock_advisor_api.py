from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import AppSettings
from .schemas import ChatRequest, SessionCreateResponse, SessionHistoryResponse, SessionListItem
from .stock_advisor_service import StockAdvisorService


def create_app() -> FastAPI:
    settings = AppSettings.from_env()
    service = StockAdvisorService(settings)
    app = FastAPI(
        title="Stock Advisor API",
        version="0.1.0",
        description="Current stock trade-plan and buy-node advisor.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        payload = service.health().model_dump()
        payload["project"] = "stock_advisor"
        return payload

    @app.post("/api/session", response_model=SessionCreateResponse)
    async def create_session() -> SessionCreateResponse:
        return SessionCreateResponse(session_id=service.create_session())

    @app.get("/api/sessions", response_model=list[SessionListItem])
    async def list_sessions() -> list[SessionListItem]:
        return [SessionListItem(**item) for item in service.list_sessions()]

    @app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
    async def get_session(session_id: str) -> SessionHistoryResponse:
        try:
            return SessionHistoryResponse(**service.get_session_history(session_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        stream = service.stream_chat(request.session_id, request.message.strip())
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


def run() -> None:
    settings = AppSettings.from_env()
    uvicorn.run(
        "invest_digital_human.stock_advisor_api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )

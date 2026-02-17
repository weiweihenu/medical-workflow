from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, TypedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents import IntakeAgent, RouterAgent, SpecialistAgent, SummaryAgent
from app.config import AppConfig
from app.llm_client import LLMClient
from app.ocr_gpt4o import extract_text_from_upload_with_gpt4o

try:
    from app.workflow import create_initial_state as _create_initial_state
except Exception:
    _create_initial_state = None


class ConsultStreamRequest(BaseModel):
    session_id: str
    user_input: str


class SessionBucket(TypedDict):
    state: Dict[str, Any]
    documents: List[Dict[str, Any]]


AgentCallable = Callable[[Dict[str, Any]], Dict[str, Any]]

app = FastAPI(title="Medical Workflow API (True Token SSE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = AppConfig.from_env()
llm = LLMClient(config)

intake_agent = IntakeAgent(llm)
router_agent = RouterAgent(llm)
specialist_agent = SpecialistAgent(llm)
summary_agent = SummaryAgent(llm)

SESSION_DB: Dict[str, SessionBucket] = {}

OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4o-mini")
OCR_MAX_PDF_PAGES = int(os.getenv("OCR_MAX_PDF_PAGES", "8"))
STAGE_HEARTBEAT_SECONDS = float(os.getenv("SSE_STAGE_HEARTBEAT_SECONDS", "0.8"))

STAGE_NAME_MAP = {
    "intake": "询问智能体",
    "router": "路由智能体",
    "specialist": "专科智能体",
    "summary_structured": "总结智能体(结构化)",
    "summary_reply": "总结智能体(流式回复)",
}

STAGE_TIPS = {
    "intake": ["正在整理主诉信息", "正在检查缺失字段", "正在生成补充提问"],
    "router": ["正在判断分诊科室", "正在比对关键症状", "正在生成分诊理由"],
    "specialist": ["正在进行专科分析", "正在整理检查建议", "正在评估风险提醒"],
    "summary_structured": ["正在整合结构化结论", "正在生成复诊与急症边界"],
}


def _new_state() -> Dict[str, Any]:
    if callable(_create_initial_state):
        state = dict(_create_initial_state())
    else:
        state = {
            "history": [],
            "patient_info": {},
            "documents_text": [],
            "next_action": "continue",
        }

    state.setdefault("history", [])
    state.setdefault("patient_info", {})
    state.setdefault("documents_text", [])
    state.setdefault("next_action", "continue")
    return state


def _get_bucket(session_id: str) -> SessionBucket:
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    if sid not in SESSION_DB:
        SESSION_DB[sid] = {"state": _new_state(), "documents": []}
    return SESSION_DB[sid]


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 12):
    if not text:
        yield ""
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _append_history(state: Dict[str, Any], role: str, content: str) -> None:
    history = state.get("history")
    if not isinstance(history, list):
        history = []
    history.append({"role": role, "content": content})
    state["history"] = history


def _build_progress_message(stage_key: str, tick: int) -> str:
    tips = STAGE_TIPS.get(stage_key, ["处理中"])
    tip = tips[tick % len(tips)]
    elapsed = int((tick + 1) * STAGE_HEARTBEAT_SECONDS)
    return f"{tip} (约 {elapsed}s)"


def _infer_case_status(next_action: str) -> str:
    if next_action == "ask_user_more":
        return "collecting"
    if next_action == "done":
        return "closed"
    return "in_progress"


def _build_state_snapshot(
    session_id: str,
    state: Dict[str, Any],
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []

    followup_round = sum(
        1 for item in history if isinstance(item, dict) and item.get("role") == "user"
    )

    route = state.get("route", {})
    if not isinstance(route, dict):
        route = {}

    next_action = str(state.get("next_action") or "continue")

    return {
        "session_id": session_id,
        "next_action": next_action,
        "case_status": _infer_case_status(next_action),
        "followup_round": followup_round,
        "route": route,
        "doc_count": len(docs),
    }


async def _run_blocking_stage(
    state: Dict[str, Any],
    stage_key: str,
    stage_callable: AgentCallable,
) -> AsyncGenerator[Dict[str, Any], None]:
    stage_name = STAGE_NAME_MAP.get(stage_key, stage_key)

    yield {
        "type": "stage_start",
        "stage": stage_key,
        "stage_name": stage_name,
        "message": "已启动",
    }

    task = asyncio.create_task(asyncio.to_thread(stage_callable, state))
    tick = 0

    while not task.done():
        yield {
            "type": "stage_progress",
            "stage": stage_key,
            "stage_name": stage_name,
            "message": _build_progress_message(stage_key, tick),
        }
        tick += 1
        await asyncio.sleep(STAGE_HEARTBEAT_SECONDS)

    update = await task
    if not isinstance(update, dict):
        raise RuntimeError(f"{stage_name} 返回结果不是 dict")

    state.update(update)

    preview = str(state.get("assistant_reply") or "").strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."

    payload: Dict[str, Any] = {
        "type": "stage_done",
        "stage": stage_key,
        "stage_name": stage_name,
        "message": "处理完成",
        "next_action": str(state.get("next_action") or "continue"),
    }
    if preview:
        payload["preview"] = preview

    if stage_key == "router":
        route = state.get("route", {})
        if isinstance(route, dict):
            payload["department"] = route.get("department", "")

    yield payload


@app.get("/api/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/api/documents/upload")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    if not files:
        raise HTTPException(status_code=400, detail="files 不能为空")

    bucket = _get_bucket(sid)
    saved_docs: List[Dict[str, Any]] = []

    for file in files:
        file_bytes = await file.read()
        filename = (file.filename or "unnamed").strip() or "unnamed"

        extracted_text = extract_text_from_upload_with_gpt4o(
            client=llm.client,
            filename_pages=OCR_MAX_PDF_PAGES,
        )
        extracted_text = str(extracted_text or "").strip()

        doc = {
            "doc_id": str(uuid.uuid4()),
            "filename": filename,
            "content_type": file.content_type or "application/octet-stream",
            "extracted_text": extracted_text,
        }
        bucket["documents"].append(doc)

        saved_docs.append(
            {
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "content_type": doc["content_type"],
                "char_count": len(extracted_text),
                "preview": extracted_text[:180],
            }
        )

    return {
        "session_id": sid,
        "documents": saved_docs,
        "total_documents": len(bucket["documents"]),
    }


@app.post("/api/consult/stream")
async def consult_stream(req: ConsultStreamRequest):
    sid = (req.session_id or "").strip()
    user_input = (req.user_input or "").strip()

    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    if not user_input:
        raise HTTPException(status_code=400, detail="user_input 不能为空")

    bucket = _get_bucket(sid)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            state = dict(bucket["state"])
            state["user_input"] = user_input
            state["documents_text"] = [
                str(d.get("extracted_text", "")).strip()
                for d in bucket["documents"]
                if str(d.get("extracted_text", "")).strip()
            ]
            _append_history(state, "user", user_input)

            yield _sse(
                {
                    "type": "meta",
                    "session_id": sid,
                    "doc_count": len(bucket["documents"]),
                }
            )

            async for evt in _run_blocking_stage(state, "intake", intake_agent):
                yield _sse(evt)

            if str(state.get("next_action") or "") == "ask_user_more":
                assistant_reply = str(state.get("assistant_reply") or "请补充更多信息。")
                for part in _chunk_text(assistant_reply, size=10):
                    yield _sse({"type": "token", "content": part})
                    await asyncio.sleep(0.01)

                bucket["state"] = state
                snapshot = _build_state_snapshot(sid, state, bucket["documents"])

                yield _sse(
                    {
                        "type": "final",
                        "session_id": sid,
                        "assistant_reply": assistant_reply,
                        "next_action": state.get("next_action", "ask_user_more"),
                        "state": snapshot,
                    }
                )
                yield "data: [DONE]\n\n"
                return

            async for evt in _run_blocking_stage(state, "router", router_agent):
                yield _sse(evt)

            async for evt in _run_blocking_stage(state, "specialist", specialist_agent):
                yield _sse(evt)

            async for evt in _run_blocking_stage(state, "summary_structured", summary_agent.prepare):
                yield _sse(evt)

            yield _sse(
                {
                    "type": "stage_start",
                    "stage": "summary_reply",
                    "stage_name": STAGE_NAME_MAP["summary_reply"],
                    "message": "开始逐 token 生成最终回复",
                }
            )

            assistant_reply = ""
            try:
                for token in summary_agent.stream_reply(state):
                    if not token:
                        continue
                    assistant_reply += token
                    yield _sse({"type": "token", "content": token})
            except Exception as exc:
                yield _sse(
                    {
                        "type": "stage_progress",
                        "stage": "summary_reply",
                        "stage_name": STAGE_NAME_MAP["summary_reply"],
                        "message": f"流式中断，使用兜底回复: {exc}",
                    }
                )

            if not assistant_reply.strip():
                fallback = summary_agent.fallback_reply(state)
                for part in _chunk_text(fallback, size=10):
                    assistant_reply += part
                    yield _sse({"type": "token", "content": part})
                    await asyncio.sleep(0.01)

            yield _sse(
                {
                    "type": "stage_done",
                    "stage": "summary_reply",
                    "stage_name": STAGE_NAME_MAP["summary_reply"],
                    "message": "流式输出完成",
                }
            )

            state["assistant_reply"] = assistant_reply
            state["next_action"] = "done"
            _append_history(state, "assistant", assistant_reply)

            bucket["state"] = state
            snapshot = _build_state_snapshot(sid, state, bucket["documents"])

            yield _sse(
                {
                    "type": "final",
                    "session_id": sid,
                    "assistant_reply": assistant_reply,
                    "next_action": state.get("next_action", "done"),
                    "state": snapshot,
                }
            )
            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield _sse({"type": "error", "message": f"工作流异常: {exc}"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

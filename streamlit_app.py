from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_URL = f"{API_BASE_URL}/api/documents/upload"
STREAM_URL = f"{API_BASE_URL}/api/consult/stream"

MAX_SILENCE_SECONDS = int(os.getenv("UI_MAX_SILENCE_SECONDS", "120"))

ANIM_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PULSE_FRAMES = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]

STARTUP_MESSAGE = (
    "你好，我是智能医疗问诊助手。\n"
    "请先描述主要症状、持续时间、严重程度。"
)


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def _build_live_status_text(
    frame_index: int,
    stage_name: str,
    stage_mode: str,
    workflow_start_at: float,
    stage_start_at: float,
    last_backend_event_at: float,
    extra_message: str = "",
) -> str:
    now = time.time()
    stage_elapsed = _format_duration(now - stage_start_at)
    total_elapsed = _format_duration(now - workflow_start_at)
    idle_elapsed = _format_duration(now - last_backend_event_at)

    spinner = ANIM_FRAMES[frame_index % len(ANIM_FRAMES)]
    pulse = PULSE_FRAMES[frame_index % len(PULSE_FRAMES)]

    text = (
        f"{spinner} {stage_name} {stage_mode} {pulse}\n\n"
        f"阶段耗时：{stage_elapsed} ｜ 总耗时：{total_elapsed} ｜ 最近后端更新：{idle_elapsed}前"
    )
    if extra_message:
        text += f"\n\n{extra_message}"
    return text


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = _new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": STARTUP_MESSAGE}]
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "state_snapshot" not in st.session_state:
        st.session_state.state_snapshot = {}


def _reset_case() -> None:
    st.session_state.session_id = _new_session_id()
    st.session_state.messages = [{"role": "assistant", "content": STARTUP_MESSAGE}]
    st.session_state.uploaded_docs = []
    st.session_state.state_snapshot = {}


def _upload_documents(files) -> Dict[str, Any]:
    multipart_files = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]

    response = requests.post(
        UPLOAD_URL,
        data={"session_id": st.session_state.session_id},
        files=multipart_files,
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"上传失败({response.status_code}): {response.text}")
    return response.json()


def _stream_worker(session_id: str, user_input: str, out_q) -> None:
    payload = {"session_id": session_id, "user_input": user_input}

    try:
        with requests.post(STREAM_URL, json=payload, stream=True, timeout=(20, 600)) as response:
            if not response.ok:
                out_q.put({"type": "error", "message": f"请求失败({response.status_code}): {response.text}"})
                return

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(event, dict):
                    out_q.put(event)

    except Exception as exc:
        out_q.put({"type": "error", "message": f"流式连接异常: {exc}"})
    finally:
        out_q.put({"type": "_worker_done"})


def _render_sidebar_snapshot(snapshot: Dict[str, Any]) -> None:
    route = snapshot.get("route", {}) if isinstance(snapshot.get("route"), dict) else {}
    st.write("next_action:", snapshot.get("next_action", "-"))
    st.write("case_status:", snapshot.get("case_status", "-"))
    st.write("followup_round:", snapshot.get("followup_round", "-"))
    st.write("department:", route.get("department", "-"))
    st.write("doc_count:", snapshot.get("doc_count", 0))


st.set_page_config(page_title="智能医疗问诊", page_icon="🩺", layout="wide")
_init_state()

st.title("🩺 智能医疗问诊")

with st.sidebar:
    st.subheader("会话")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    if st.button("🆕 新病例", use_container_width=True):
        _reset_case()
        st.rerun()

    st.divider()
    st.subheader("上传检查单 / 纸质材料")

    upload_files = st.file_uploader(
        "支持 png/jpg/jpeg/pdf/txt/md",
        type=["png", "jpg", "jpeg", "pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("📤 上传并解析", use_container_width=True, disabled=not upload_files):
        try:
            result = _upload_documents(upload_files)
            docs = result.get("documents", [])
            if isinstance(docs, list):
                st.session_state.uploaded_docs.extend(docs)
            st.success(f"上传成功: 新增 {len(docs)} 份材料")
        except Exception as exc:
            st.error(f"上传失败: {exc}")

    if st.session_state.uploaded_docs:
        st.markdown("**已接入材料**")
        for doc in st.session_state.uploaded_docs[-10:]:
            filename = doc.get("filename", "unnamed")
            char_count = doc.get("char_count", 0)
            st.caption(f"- {filename} ({char_count} 字)")

    st.divider()
    st.subheader("状态快照")
    _render_sidebar_snapshot(st.session_state.state_snapshot)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("请输入症状、持续时间、严重程度...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    assistant_text = ""
    final_state: Dict[str, Any] = {}

    with st.chat_message("assistant"):
        status_box = st.empty()
        answer_box = st.empty()

        workflow_start_at = time.time()
        current_stage_name = "系统连接"
        current_stage_start_at = workflow_start_at
        last_backend_event_at = workflow_start_at
        stage_mode = "准备中"
        stage_extra = "正在建立流式连接..."

        spinner_index = 0
        runtime_error = ""

        worker_queue = queue.Queue()
        worker = threading.Thread(
            target=_stream_worker,
            args=(st.session_state.session_id, user_text, worker_queue),
            daemon=True,
        )
        worker.start()

        status_box.info(
            _build_live_status_text(
                frame_index=spinner_index,
                stage_name=current_stage_name,
                stage_mode=stage_mode,
                workflow_start_at=workflow_start_at,
                stage_start_at=current_stage_start_at,
                last_backend_event_at=last_backend_event_at,
                extra_message=stage_extra,
            )
        )

        stream_done = False
        while not stream_done:
            try:
                event = worker_queue.get(timeout=0.20)
            except queue.Empty:
                if (time.time() - last_backend_event_at) > MAX_SILENCE_SECONDS:
                    runtime_error = f"后端超过 {MAX_SILENCE_SECONDS} 秒无更新，请稍后重试"
                    break

                spinner_index += 1
                status_box.info(
                    _build_live_status_text(
                        frame_index=spinner_index,
                        stage_name=current_stage_name,
                        stage_mode=stage_mode,
                        workflow_start_at=workflow_start_at,
                        stage_start_at=current_stage_start_at,
                        last_backend_event_at=last_backend_event_at,
                        extra_message=stage_extra,
                    )
                )
                continue

            event_type = str(event.get("type", ""))

            if event_type == "_worker_done":
                stream_done = True
                continue

            last_backend_event_at = time.time()

            if event_type == "error":
                runtime_error = str(event.get("message", "未知错误"))
                break

            if event_type == "meta":
                doc_count = event.get("doc_count", 0)
                current_stage_name = "会话初始化"
                stage_mode = "已连接"
                stage_extra = f"已连接后端，当前接入材料 {doc_count} 份"
                spinner_index += 1
                status_box.info(
                    _build_live_status_text(
                        frame_index=spinner_index,
                        stage_name=current_stage_name,
                        stage_mode=stage_mode,
                        workflow_start_at=workflow_start_at,
                        stage_start_at=current_stage_start_at,
                        last_backend_event_at=last_backend_event_at,
                        extra_message=stage_extra,
                    )
                )
                continue

            if event_type in {"stage_start", "stage_progress", "stage_done"}:
                stage_name = str(event.get("stage_name", "智能体"))
                message = str(event.get("message", "")).strip()

                if event_type == "stage_start":
                    current_stage_name = stage_name
                    current_stage_start_at = time.time()
                    stage_mode = "启动中"
                    stage_extra = message or "阶段已启动"
                    spinner_index += 1
                    status_box.info(
                        _build_live_status_text(
                            frame_index=spinner_index,
                            stage_name=current_stage_name,
                            stage_mode=stage_mode,
                            workflow_start_at=workflow_start_at,
                            stage_start_at=current_stage_start_at,
                            last_backend_event_at=last_backend_event_at,
                            extra_message=stage_extra,
                        )
                    )
                elif event_type == "stage_progress":
                    current_stage_name = stage_name
                    stage_mode = "处理中"
                    stage_extra = message or "正在处理中"
                    spinner_index += 1
                    status_box.info(
                        _build_live_status_text(
                            frame_index=spinner_index,
                            stage_name=current_stage_name,
                            stage_mode=stage_mode,
                            workflow_start_at=workflow_start_at,
                            stage_start_at=current_stage_start_at,
                            last_backend_event_at=last_backend_event_at,
                            extra_message=stage_extra,
                        )
                    )
                else:
                    stage_elapsed = _format_duration(time.time() - current_stage_start_at)
                    total_elapsed = _format_duration(time.time() - workflow_start_at)
                    done_tip = message or "处理完成"
                    status_box.success(
                        f"✅ {stage_name} 完成（阶段 {stage_elapsed} / 总计 {total_elapsed}）\n\n{done_tip}"
                    )
                    current_stage_name = "等待下一阶段"
                    current_stage_start_at = time.time()
                    stage_mode = "排队中"
                    stage_extra = "上一阶段已完成，准备进入下一阶段..."
                continue

            if event_type == "token":
                token = str(event.get("content", ""))
                if token:
                    assistant_text += token
                    answer_box.markdown(assistant_text)

                current_stage_name = "总结回复"
                stage_mode = "输出中"
                stage_extra = "正在逐字生成答案..."
                continue

            if event_type == "final":
                final_reply = str(event.get("assistant_reply", "")).strip()
                if final_reply:
                    assistant_text = final_reply
                    answer_box.markdown(assistant_text)

                payload_state = event.get("state")
                if isinstance(payload_state, dict):
                    final_state = payload_state

                current_stage_name = "结果收尾"
                stage_mode = "完成中"
                stage_extra = "正在保存本轮状态..."
                continue

        if worker.is_alive():
            worker.join(timeout=1.0)

        total_elapsed = _format_duration(time.time() - workflow_start_at)

        if runtime_error:
            if assistant_text.strip():
                assistant_text = f"{assistant_text}\n\n（注意：{runtime_error}）"
            else:
                assistant_text = f"请求失败: {runtime_error}"
            status_box.error(f"❌ 本轮处理失败（总耗时 {total_elapsed}）")
            answer_box.markdown(assistant_text)
        else:
            if not assistant_text.strip():
                assistant_text = "抱歉，本轮没有生成有效回复。"
                answer_box.markdown(assistant_text)

            status_box.success(f"✅ 本轮处理完成（总耗时 {total_elapsed}）")

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    if final_state:
        st.session_state.state_snapshot = final_state

    next_action = str(st.session_state.state_snapshot.get("next_action", ""))
    if next_action == "ask_user_more":
        st.warning("需要补充信息后再继续。")
    elif next_action == "done":
        st.info("本轮问诊已完成，你可以继续追问或点击“新病例”。")

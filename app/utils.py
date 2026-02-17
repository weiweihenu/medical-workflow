from __future__ import annotations

from typing import Any, Dict, List

from app.state import Department, FinalResult, PatientInfo


def history_to_text(history: List[Dict[str, str]], limit: int = 12) -> str:
    """将最近 N 条历史消息转成字符串，供 LLM 理解上下文。"""
    lines: List[str] = []
    for msg in history[-limit:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def as_str_list(value: Any) -> List[str]:
    """将任意值安全转换为字符串列表，并去重。"""
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def merge_patient_info(old_info: PatientInfo, new_info: Dict[str, Any]) -> PatientInfo:
    """
    合并 patient_info：
    - 标量字段：新值覆盖旧值（前提是新值非空）
    - 列表字段：去重合并
    """
    merged: Dict[str, Any] = dict(old_info or {})
    for key, value in (new_info or {}).items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            old_values = as_str_list(merged.get(key, []))
            for item in as_str_list(value):
                if item not in old_values:
                    old_values.append(item)
            merged[key] = old_values
        else:
            merged[key] = value
    return merged  # type: ignore[return-value]


def route_fallback(patient_info: Dict[str, Any]) -> Department:
    """
    当路由智能体输出异常时，使用关键词兜底路由。
    """
    text_parts = [
        str(patient_info.get("chief_complaint", "")),
        str(patient_info.get("additional_notes", "")),
        " ".join(as_str_list(patient_info.get("symptoms", []))),
    ]
    text = " ".join(text_parts).lower()

    if any(k in text for k in ["咳", "喘", "呼吸", "痰", "气短", "胸闷"]):
        return "呼吸科"
    if any(k in text for k in ["胸痛", "心悸", "心慌", "心率", "血压", "心口"]):
        return "心血管科"
    if any(k in text for k in ["腹痛", "腹泻", "反酸", "胃痛", "恶心", "呕吐"]):
        return "消化内科"
    return "全科"


def format_bullets(items: List[str]) -> str:
    """把列表渲染成 markdown 风格条目。"""
    if not items:
        return "- 暂无"
    return "\n".join(f"- {item}" for item in items)


def render_final_reply(department: str, final_result: FinalResult) -> str:
    """把总结智能体的结构化 JSON 转换成用户可读文本。"""
    diagnosis_summary = final_result.get("diagnosis_summary", "")
    prescription_advice = as_str_list(final_result.get("prescription_advice", []))
    home_care = as_str_list(final_result.get("home_care", []))
    follow_up = as_str_list(final_result.get("follow_up", []))
    emergency_signs = as_str_list(final_result.get("emergency_signs", []))
    disclaimer = final_result.get("disclaimer", "本结果仅作健康参考，不替代医生面诊与处方。")

    return (
        "【问诊总结】\n"
        f"分诊科室：{department}\n"
        f"初步判断：{diagnosis_summary}\n\n"
        "【建议用药（仅供与医生讨论）】\n"
        f"{format_bullets(prescription_advice)}\n\n"
        "【居家护理】\n"
        f"{format_bullets(home_care)}\n\n"
        "【复诊建议】\n"
        f"{format_bullets(follow_up)}\n\n"
        "【立即就医信号】\n"
        f"{format_bullets(emergency_signs)}\n\n"
        f"【声明】{disclaimer}"
    )

def build_document_context(documents_text: List[str], max_chars: int = 3000) -> str:
    cleaned = [t.strip() for t in documents_text if isinstance(t, str) and t.strip()]
    if not cleaned:
        return "无上传参考材料"
    merged = "\n\n".join([f"参考材料{i+1}:\n{text}" for i, text in enumerate(cleaned)])
    return merged[:max_chars]
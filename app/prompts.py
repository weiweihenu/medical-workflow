from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def _to_json(value: Any) -> str:
    """统一 JSON 序列化（保留中文）。"""
    return json.dumps(value, ensure_ascii=False)


def _normalize_doc_context(document_context: str) -> str:
    """保证上传材料上下文稳定可用。"""
    text = str(document_context or "").strip()
    return text if text else "无上传参考材料"


def build_intake_prompts(
    history_text: str,
    existing_info: Dict[str, Any],
    document_context: str = "无上传参考材料",
) -> Tuple[str, str]:
    """
    询问智能体提示词：
    - 只负责采集信息，不做诊断、不做开药
    - 输出严格 JSON
    """
    doc_context = _normalize_doc_context(document_context)

    system_prompt = (
        "你是“询问智能体（Intake Agent）”。"
        "你的职责是采集和补全问诊信息，不做诊断、不做开药。"
        "你必须只输出严格 JSON，不允许输出 JSON 之外的任何文字。"
    )

    user_prompt = (
        "请根据“历史对话 + 既有病历 + 上传参考材料”抽取并更新 patient_info。\n"
        "输出 JSON（严格按此结构）：\n"
        "{\n"
        '  "patient_info": {\n'
        '    "age": null,\n'
        '    "sex": null,\n'
        '    "chief_complaint": "",\n'
        '    "duration": "",\n'
        '    "severity": "",\n'
        '    "symptoms": [],\n'
        '    "allergies": [],\n'
        '    "chronic_diseasesn'
        '    "additional_notes": ""\n'
        "  },\n"
        '  "is_complete": true,\n'
        '  "missing_fields": [],\n'
        '  "missing_questions": []\n'
        "}\n\n"
        "规则：\n"
        "1) 只做信息采集，不要给诊断结论，不要给药物建议。\n"
        "2) is_complete=true 的条件：chief_complaint、duration、severity 三项都有明确内容。\n"
        "3) missing_fields 只能从以下字段中选择："
        "age, sex, chief_complaint, duration, severity, symptoms, allergies, chronic_diseases, current_meds。\n"
        "4) missing_questions 必须是给患者看的中文完整问题，且尽量包含示例；"
        "禁止只输出英文键名（例如 severity）。\n"
        "5) 若上传材料与用户最新描述冲突，以用户最新描述为准；"
        "可在 additional_notes 中记录冲突信息。\n"
        "6) 信息不确定时不要猜测，保持为空。\n\n"
        f"existing_info = {_to_json(existing_info)}\n"
        f"history = \n{history_text}\n"
        f"uploaded_reference = {_to_json(doc_context)}"
    )
    return system_prompt, user_prompt


def build_router_prompts(
    patient_info: Dict[str, Any],
    document_context: str = "无上传参考材料",
) -> Tuple[str, str]:
    """
    路由智能体提示词：
    - 在 呼吸科 / 心血管科 / 消化内科 / 全科 中选择一个
    - 输出严格 JSON
    """
    doc_context = _normalize_doc_context(document_context)

    system_prompt = (
        "你是“路由智能体（Router Agent）”。"
        "你需要根据患者信息分诊到一个最合适的科室："
        "呼吸科、心血管科、消化内科、全科。"
        "你必须只输出严格 JSON。"
    )

    user_prompt = (
        "请输出 JSON：\n"
        "{\n"
        '  "department": "呼吸科|心血管科|消化内科|全科",\n'
        '  "reason": "",\n'
        '  "confidence": 0.0,\n'
        '  "key_evidence": []\n'
        "}\n\n"
        "分诊规则：\n"
        "1) 咳嗽、咳痰、气短、喘息等呼吸道症状为主时，优先呼吸科。\n"
        "2) 胸痛、心悸、心慌、血压异常等心血管症状为主时，优先心血管科。\n"
        "3) 腹痛、腹泻、反酸、恶心、呕吐等消化症状为主时，优先消化内科。\n"
        "4) 信息不足或症状交叉明显时，路由到全科。\n"
        "5) 若上传材料与用户最新主诉冲突，以用户最新主诉为准。\n"
        "6) reason 请用中文写清楚，便于患者理解。\n\n"
        f"patient_info = {_to_json(patient_info)}\n"
        f"uploaded_reference = {_to_json(doc_context)}"
    )
    return system_prompt, user_prompt


def build_specialist_prompts(
    department: str,
    patient_info: Dict[str, Any],
    route: Dict[str, Any],
    document_context: str = "无上传参考材料",
) -> Tuple[str, str]:
    """
    专科智能体提示词：
    - 给出初步分析、可能诊断方向、检查建议、可讨论的用药方向
    - 不给具体处方剂量/频次/疗程
    """
    doc_context = _normalize_doc_context(document_context)

    system_prompt = (
        f"你是“{department}专科智能体（Specialist Agent）”。"
        "请给出初步分析、可能方向、建议检查、可讨论的常见用药方向。"
        "不要给出确定性诊断；不要给出药物剂量、频次、疗程。"
        "你必须只输出严格 JSON。"
    )

    user_prompt = (
        "请输出 JSON：\n"
        "{\n"
        '  "preliminary_assessment": "",\n'
        '  "possible_diagnoses": [],\n'
        '  "recommended_checks": [],\n'
        '  "medication_suggestions": [\n'
        '    {"name": "", "purpose": "", "otc": true}\n'
        "  ],\n"
        '  "risk_alerts": []\n'
        "}\n\n"
        "要求：\n"
        "1) preliminary_assessment 用中文，面向患者可读。\n"
        "2) possible_diagnoses 只写“可能方向”，不要写“已确诊”。\n"
        "3) medication_suggestions 仅写“可与医生讨论的药物方向”，"
        "禁止剂量、频次、疗程。\n"
        "4) risk_alerts 必n"
        "4) risk_alerts 必须包含就医边界（何时应立即就医）。\n"
        "5) 若上传材料与用户最新主诉冲突，以用户最新主诉为准。\n\n"
        f"patient_info = {_to_json(patient_info)}\n"
        f"route = {_to_json(route)}\n"
        f"uploaded_reference = {_to_json(doc_context)}"
    )
    return system_prompt, user_prompt


def build_summary_prompts(
    patient_info: Dict[str, Any],
    route: Dict[str, Any],
    specialist_result: Dict[str, Any],
    document_context: str = "无上传参考材料",
) -> Tuple[str, str]:
    """
    总结智能体提示词：
    - 输出患者可读的最终总结
    - 必须包含安全边界声明
    """
    doc_context = _normalize_doc_context(document_context)

    system_prompt = (
        "你是“开药诊断总结智能体（Summary Agent）”。"
        "请将现有信息整合为患者可读总结，必须包含安全提醒和就医边界。"
        "你必须只输出严格 JSON。"
    )

    user_prompt = (
        "请输出 JSON：\n"
        "{\n"
        '  "diagnosis_summary": "",\n'
        '  "prescription_advice": [],\n'
        '  "home_care": [],\n'
        '  "follow_up": [],\n'
        '  "emergency_signs": [],\n'
        '  "disclaimer": ""\n'
        "}\n\n"
        "要求：\n"
        "1) diagnosis_summary 用中文简明描述“当前初步判断”。\n"
        "2) prescription_advice 仅写“可与医生讨论”的建议，"
        "禁止出现具体处方剂量、频次、疗程。\n"
        "3) home_care 写清楚可执行的居家护理建议。\n"
        "4) follow_up 写明复诊时机（例如 2-3 天无缓解或加重时复诊）。\n"
        "5) emergency_signs 写明立即就医信号（例如持续胸痛、呼吸困难、意识改变）。\n"
        "6) disclaimer 必须明确：不能替代医生面诊与处方。\n"
        "7) 若上传材料与用户最新主诉冲突，以用户最新主诉为准。\n\n"
        f"patient_info = {_to_json(patient_info)}\n"
        f"route = {_to_json(route)}\n"
        f"specialist_result = {_to_json(specialist_result)}\n"
        f"uploaded_reference = {_to_json(doc_context)}"
    )
    return system_prompt, user_prompt

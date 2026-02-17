from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Tuple

from app import prompts as prompt_builders
from app.llm_client import LLMClient

try:
    from app.utils import render_final_reply, route_fallback
except Exception:
    def route_fallback(_: Dict[str, Any]) -> str:
        return "全科"

    def render_final_reply(department: str, final_result: Dict[str, Any]) -> str:
        diagnosis = str((final_result or {}).get("diagnosis_summary") or "当前为初步健康评估。")
        return f"【初步判断】\n{diagnosis}\n\n【建议科室】\n{department}"


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _to_str_list(value: Any, max_items: int = 20) -> List[str]:
    if not isinstance(value, list):
        return []

    result: List[str] = []
    for item in value:
        text = _safe_text(item)
        if text and text not in result:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _to_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_safe_text(item) for item in value)
    return True


def _build_document_context(documents_text: Any, max_chars: int = 3000) -> str:
    if not isinstance(documents_text, list):
        return "无上传参考材料"

    blocks: List[str] = []
    for idx, item in enumerate(documents_text, start=1):
        text = _safe_text(item)
        if text:
            blocks.append(f"参考材料{idx}:\n{text}")

    if not blocks:
        return "无上传参考材料"

    merged = "\n\n".join(blocks)
    return merged[:max_chars]


def _history_to_text(history: List[Dict[str, Any]], user_input: str, max_turns: int = 20) -> str:
    lines: List[str] = []
    sliced = history[-max_turns:] if isinstance(history, list) else []

    for item in sliced:
        if not isinstance(item, dict):
            continue
        role = _safe_text(item.get("role"), "unknown")
        content = _safe_text(item.get("content"))
        if content:
            lines.append(f"{role}: {content}")

    append_user = bool(user_input)
    if append_user and sliced:
        last = sliced[-1]
        if isinstance(last, dict):
            if _safe_text(last.get("role")) == "user" and _safe_text(last.get("content")) == user_input:
                append_user = False

    if append_user:
        lines.append(f"user: {user_input}")

    return "\n".join(lines).strip()


def _has_red_flag(patient_info: Dict[str, Any]) -> bool:
    text_parts = [
        _safe_text(patient_info.get("chief_complaint")),
        _safe_text(patient_info.get("additional_notes")),
        " ".join(_to_str_list(patient_info.get("symptoms"))),
    ]
    merged_text = " ".join(text_parts).lower()
    red_flags = ["胸痛", "呼吸困难", "意识不清", "昏迷", "抽搐", "咯血", "高热不退"]
    return any(flag in merged_text for flag in red_flags)


class BaseAgent(ABC):
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(state)

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class IntakeAgent(BaseAgent):
    REQUIRED_FIELDS = ("chief_complaint", "duration", "severity")
    LIST_FIELDS = ("symptoms", "allergies", "chronic_diseases", "current_meds")
    SCALAR_FIELDS = ("age", "sex", "chief_complaint", "duration", "severity", "additional_notes")
    KNOWN_FIELDS = {
        "age",
        "sex",
        "chief_complaint",
        "duration",
        "severity",
        "symptoms",
        "allergies",
        "chronic_diseases",
        "current_meds",
        "additional_notes",
    }

    @staticmethod
    def _build_prompts(
        history_text: str,
        existing_info: Dict[str, Any],
        document_context: str,
    ) -> Tuple[str, str]:
        try:
            return prompt_builders.build_intake_prompts(
                history_text=history_text,
                existing_info=existing_info,
                document_context=document_context,
            )
        except TypeError:
            return prompt_builders.build_intake_prompts(history_text, existing_info)

    def _normalize_patient_info(
        self,
        raw_patient_info: Any,
        existing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "age": None,
            "sex": "",
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "symptoms": [],
            "allergies": [],
            "chronic_diseases": [],
            "current_meds": [],
            "additional_notes": "",
        }

        if isinstance(existing_info, dict):
            merged.update(existing_info)

        source = raw_patient_info if isinstance(raw_patient_info, dict) else {}

        if "age" in source:
            age_raw = source.get("age")
            age_value: Any = None
            if isinstance(age_raw, (int, float)) and age_raw >= 0:
                age_value = int(age_raw)
            else:
                age_text = _safe_text(age_raw)
                if age_text:
                    try:
                        age_value = int(float(age_text))
                    except Exception:
                        age_value = age_text
            if age_value is not None:
                merged["age"] = age_value

        for field in ("sex", "chief_complaint", "duration", "severity", "additional_notes"):
            if field in source:
                merged[field] = _safe_text(source.get(field))

        for field in self.LIST_FIELDS:
            if field in source:
                merged[field] = _to_str_list(source.get(field))
            else:
                merged[field] = _to_str_list(merged.get(field))

        for field in ("sex", "chief_complaint", "duration", "severity", "additional_notes"):
            merged[field] = _safe_text(merged.get(field))

        return merged

    def _normalize_missing_fields(self, value: Any, patient_info: Dict[str, Any]) -> List[str]:
        fields = [x for x in _to_str_list(value, max_items=20) if x in self.KNOWN_FIELDS]
        for required in self.REQUIRED_FIELDS:
            if not _has_value(patient_info.get(required)) and required not in fields:
                fields.append(required)
        return fields

    def _is_required_complete(self, patient_info: Dict[str, Any]) -> bool:
        return all(_has_value(patient_info.get(field)) for field in self.REQUIRED_FIELDS)

    @staticmethod
    def _default_questions(missing_fields: List[str]) -> List[str]:
        question_map = {
            "age": "请问您的年龄是多少岁？",
            "sex": "请问您的性别是？",
            "chief_complaint": "请您用一句话描述最主要的不舒服是什么？",
            "duration": "这些症状持续了多久？例如 2 天、1 周、1 个月。",
            "severity": "目前严重程度如何？可用轻/中/重，或 0-10 分描述。",
            "symptoms": "除了主要不适，还有哪些伴随症状？",
            "allergies": "您是否有药物或食物过敏史？",
            "chronic_diseases": "是否有高血压、糖尿病等慢性病史？",
            "current_meds": "目前正在使用哪些药物或保健品？",
            "additional_notes": "还有其他你觉得重要的补充信息吗？",
        }

        questions: List[str] = []
        for field in missing_fields:
            question = question_map.get(field)
            if question and question not in questions:
                questions.append(question)

        return questions[:8]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
        existing_info = state.get("patient_info", {}) if isinstance(state.get("patient_info"), dict) else {}
        user_input = _safe_text(state.get("user_input"))
        document_context = _build_document_context(state.get("documents_text", []))
        history_text = _history_to_text(history, user_input)

        system_prompt, user_prompt = self._build_prompts(
            history_text=history_text,
            existing_info=existing_info,
            document_context=document_context,
        )
        raw_result = self.llm.chat_json(system_prompt, user_prompt) or {}

        patient_info = self._normalize_patient_info(raw_result.get("patient_info"), existing_info)
        missing_fields = self._normalize_missing_fields(raw_result.get("missing_fields"), patient_info)
        missing_questions = _to_str_list(raw_result.get("missing_questions"), max_items=8)

        required_complete = self._is_required_complete(patient_info)
        model_complete = _as_bool(raw_result.get("is_complete"), default=False)
        is_complete = required_complete and (model_complete or not missing_fields)

        if is_complete:
            missing_fields = []
            missing_questions = []
            assistant_reply = "收到，信息已基本完整，正在进入分诊与专科分析。"
            next_action = "continue"
        else:
            if not missing_fields:
                missing_fields = self._normalize_missing_fields([], patient_info)
            if not missing_questions:
                missing_questions = self._default_questions(missing_fields)
            if not missing_questions:
                missing_questions = ["请补充主要不适、持续时间和严重程度。"]

            lines = [f"{index + 1}. {question}" for index, question in enumerate(missing_questions)]
            assistant_reply = "为了更准确地判断，请补充以下信息：\n" + "\n".join(lines)
            next_action = "ask_user_more"

        history.append({"role": "assistant", "content": assistant_reply})

        return {
            "history": history,
            "patient_info": patient_info,
            "intake_result": {
                "is_complete": is_complete,
                "missing_fields": missing_fields,
                "missing_questions": missing_questions,
            },
            "assistant_reply": assistant_reply,
            "next_action": next_action,
        }


class RouterAgent(BaseAgent):
    VALID_DEPARTMENTS = {"呼吸科", "心血管科", "消化内科", "全科"}

    @staticmethod
    def _build_prompts(patient_info: Dict[str, Any], document_context: str) -> Tuple[str, str]:
        try:
            return prompt_builders.build_router_prompts(
                patient_info=patient_info,
                document_context=document_context,
            )
        except TypeError:
            return prompt_builders.build_router_prompts(patient_info)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
        patient_info = state.get("patient_info", {}) if isinstance(state.get("patient_info"), dict) else {}
        document_context = _build_document_context(state.get("documents_text", []))

        system_prompt, user_prompt = self._build_prompts(patient_info, document_context)
        raw_result = self.llm.chat_json(system_prompt, user_prompt) or {}

        department = _safe_text(raw_result.get("department"))
        reason = _safe_text(raw_result.get("reason"))
        key_evidence = _to_str_list(raw_result.get("key_evidence"), max_items=8)

        confidence = _to_optional_float(raw_result.get("confidence"))
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))

        if department not in self.VALID_DEPARTMENTS:
            department = route_fallback(patient_info)
            if not reason:
                reason = "模型分诊结果不稳定，已使用规则兜底。"

        if not reason:
            reason = "依据主要症状与病程信息进行分诊。"

        if document_context != "无上传参考材料" and "上传材料" not in reason:
            reason = f"{reason}（已参考上传材料）"

        route: Dict[str, Any] = {
            "department": department,
            "reason": reason,
            "key_evidence": key_evidence,
        }
        if confidence is not None:
            route["confidence"] = confidence

        assistant_reply = f"分诊结果：{department}。理由：{reason}。正在进入专科分析。"
        history.append({"role": "assistant", "content": assistant_reply})

        return {
            "history": history,
            "route": route,
            "assistant_reply": assistant_reply,
            "next_action": "continue",
        }


class SpecialistAgent(BaseAgent):
    @staticmethod
    def _build_prompts(
        department: str,
        patient_info: Dict[str, Any],
        route: Dict[str, Any],
        document_context: str,
    ) -> Tuple[str, str]:
        try:
            return prompt_builders.build_specialist_prompts(
                department=department,
                patient_info=patient_info,
                route=route,
                document_context=document_context,
            )
        except TypeError:
            return prompt_builders.build_specialist_prompts(department, patient_info, route)

    def _normalize_specialist_result(
        self,
        raw_result: Dict[str, Any],
        patient_info: Dict[str, Any],
        document_context: str,
    ) -> Dict[str, Any]:
        if not isinstance(raw_result, dict):
            raw_result = {}

        preliminary_assessment = _safe_text(
            raw_result.get("preliminary_assessment"),
            "当前信息不足，建议线下面诊并完善检查。",
        )
        possible_diagnoses = _to_str_list(raw_result.get("possible_diagnoses"))
        recommended_checks = _to_str_list(raw_result.get("recommended_checks"))
        risk_alerts = _to_str_list(raw_result.get("risk_alerts"))

        medication_suggestions: List[Dict[str, Any]] = []
        raw_meds = raw_result.get("medication_suggestions", [])
        if isinstance(raw_meds, list):
            for item in raw_meds:
                if not isinstance(item, dict):
                    continue
                name = _safe_text(item.get("name"), "未命名药物")
                purpose = _safe_text(item.get("purpose"), "用于对症支持")
                otc = bool(item.get("otc", True))
                medication_suggestions.append(
                    {"name": name, "purpose": purpose, "otc": otc}
                )

        if recommended_checks:
            recommended_checks = ["血常规", "必要时影像学检查"]
        if not risk_alerts:
            risk_alerts = ["若症状持续或加重，请及时线下就医。"]

        if _has_red_flag(patient_info):
            urgent_tip = "若出现持续胸痛、呼吸困难或意识改变，请立即急诊就医。"
            if urgent_tip not in risk_alerts:
                risk_alerts.append(urgent_tip)

        if document_context != "无上传参考材料" and "已参考上传材料" not in preliminary_assessment:
            preliminary_assessment = f"{preliminary_assessment}（已参考上传材料）"

        return {
            "preliminary_assessment": preliminary_assessment,
            "possible_diagnoses": possible_diagnoses,
            "recommended_checks": recommended_checks,
            "medication_suggestions": medication_suggestions,
            "risk_alerts": risk_alerts,
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
        patient_info = state.get("patient_info", {}) if isinstance(state.get("patient_info"), dict) else {}
        route = state.get("route", {}) if isinstance(state.get("route"), dict) else {}
        department = _safe_text(route.get("department"), "全科")
        document_context = _build_document_context(state.get("documents_text", []))

        system_prompt, user_prompt = self._build_prompts(
            department=department,
            patient_info=patient_info,
            route=route,
            document_context=document_context,
        )
        raw_result = self.llm.chat_json(system_prompt, user_prompt) or {}
        specialist_result = self._normalize_specialist_result(raw_result, patient_info, document_context)

        assistant_reply = f"{department}专科已完成初步评估，正在生成最终总结。"
        history.append({"role": "assistant", "content": assistant_reply})

        return {
            "history": history,
            "specialist_result": specialist_result,
            "assistant_reply": assistant_reply,
            "next_action": "continue",
        }


class SummaryAgent(BaseAgent):
    @staticmethod
    def _build_summary_prompts(
        patient_info: Dict[str, Any],
        route: Dict[str, Any],
        specialist_result: Dict[str, Any],
        document_context: str,
    ) -> Tuple[str, str]:
        try:
            return prompt_builders.build_summary_prompts(
                patient_info=patient_info,
                route=route,
                specialist_result=specialist_result,
                document_context=document_context,
            )
        except TypeError:
            return prompt_builders.build_summary_prompts(patient_info, route, specialist_result)

    @staticmethod
    def _build_summary_reply_prompts_fallback(
        patient_info: Dict[str, Any],
        route: Dict[str, Any],
        specialist_result: Dict[str, Any],
        final_result: Dict[str, Any],
        document_context: str,
    ) -> Tuple[str, str]:
        system_prompt = (
            "你是医疗问诊总结助手。"
            "请基于结构化信息输出最终患者可读回复。"
            "要求：中文、清晰、可执行；不要输出JSON；"
            "不要给药物剂量/频次/疗程。"
        )

        payload = {
            "patient_info": patient_info,
            "route": route,
            "specialist_result": specialist_result,
            "final_result": final_result,
            "uploaded_reference": document_context,
        }
        user_prompt = (
            "请按以下结构输出：\n"
            "【初步判断】\n【可与医生讨论的用药方向】\n【居家护理】\n"
            "【复诊建议】\n【立即就医信号】\n【免责声明】\n\n"
            "如上传材料与用户最新主诉冲突，以最新主诉为准。\n\n"
            f"结构化输入：{json.dumps(payload, ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def _normalize_final_result(
        self,
        raw_result: Dict[str, Any],
        patient_info: Dict[str, Any],
        document_context: str,
    ) -> Dict[str, Any]:
        if not isinstance(raw_result, dict):
            raw_result = {}

        diagnosis_summary = _safe_text(
            raw_result.get("diagnosis_summary"),
            "当前结果为初步健康评估，建议结合线下面诊进一步明确。",
        )
        prescription_advice = _to_str_list(raw_result.get("prescription_advice"))
        home_care = _to_str_list(raw_result.get("home_care"))
        follow_up = _to_str_list(raw_result.get("follow_up"))
        emergency_signs = _to_str_list(raw_result.get("emergency_signs"))
        disclaimer = _safe_text(
            raw_result.get("disclaimer"),
            "本结果仅作健康参考，不替代医生面诊与处方。",
        )

        if not prescription_advice:
            prescription_advice = ["可与医生讨论是否需要对症药物。"]
        if not home_care:
            home_care = ["规律休息", "补充水分", "清淡饮食", "观察症状变化"]
        if not follow_up:
            follow_up = ["若 2-3 天无缓解或症状加重，请及时复诊。"]
        if not emergency_signs:
            emergency_signs = ["持续胸痛", "呼吸困难", "意识改变", "高热不退"]

        if _has_red_flag(patient_info):
            urgent_tip = "若出现持续胸痛或呼吸困难加重，请立即急诊就医。"
            if urgent_tip not in emergency_signs:
                emergency_signs.append(urgent_tip)

        if document_context != "无上传参考材料" and "已参考上传材料" not in diagnosis_summary:
            diagnosis_summary = f"{diagnosis_summary}（已参考上传材料）"

        return {
            "diagnosis_summary": diagnosis_summary,
            "prescription_advice": prescription_advice,
            "home_care": home_care,
            "follow_up": follow_up,
            "emergency_signs": emergency_signs,
            "disclaimer": disclaimer,
        }

    def prepare(self, state: Dict[str, Any]) -> Dict[str, Any]:
        patient_info = state.get("patient_info", {}) if isinstance(state.get("patient_info"), dict) else {}
        route = state.get("route", {}) if isinstance(state.get("route"), dict) else {}
        specialist_result = (
            state.get("specialist_result", {}) if isinstance(state.get("specialist_result"), dict) else {}
        )
        document_context = _build_document_context(state.get("documents_text", []))

        system_prompt, user_prompt = self._build_summary_prompts(
            patient_info=patient_info,
            route=route,
            specialist_result=specialist_result,
            document_context=document_context,
        )
        raw_result = self.llm.chat_json(system_prompt, user_prompt) or {}
        final_result = self._normalize_final_result(raw_result, patient_info, document_context)

        return {
            "final_result": final_result,
            "next_action": "done",
        }

    def stream_reply(self, state: Dict[str, Any]) -> Generator[str, None, None]:
        patient_info = state.get("patient_info", {}) if isinstance(state.get("patient_info"), dict) else {}
        route = state.get("route", {}) if isinstance(state.get("route"), dict) else {}
        specialist_result = (
            state.get("specialist_result", {}) if isinstance(state.get("specialist_result"), dict) else {}
        )
        final_result = state.get("final_result", {}) if isinstance(state.get("final_result"), dict) else {}
        document_context = _build_document_context(state.get("documents_text", []))

        summary_reply_builder = getattr(prompt_builders, "build_summary_reply_prompts", None)

        if callable(summary_reply_builder):
            try:
                system_prompt, user_prompt = summary_reply_builder(
                    patient_info=patient_info,
                    route=route,
                    specialist_result=specialist_result,
                    final_result=final_result,
                    document_context=document_context,
                )
            except TypeError:
                system_prompt, user_prompt = summary_reply_builder(
                    patient_info,
                    route,
                    specialist_result,
                    final_result,
                )
        else:
            system_prompt, user_prompt = self._build_summary_reply_prompts_fallback(
                patient_info=patient_info,
                route=route,
                specialist_result=specialist_result,
                final_result=final_result,
                document_context=document_context,
            )

        yield from self.llm.chat_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1200,
        )

    def fallback_reply(self, state: Dict[str, Any]) -> str:
        route = state.get("route", {}) if isinstance(state.get("route"), dict) else {}
        department = _safe_text(route.get("department"), "全科")
        final_result = state.get("final_result", {}) if isinstance(state.get("final_result"), dict) else {}

        try:
            return render_final_reply(department, final_result)
        except Exception:
            diagnosis = _safe_text(final_result.get("diagnosis_summary"), "当前为初步健康评估。")
            follow_up = _to_str_list(final_result.get("follow_up"))
            emergency = _to_str_list(final_result.get("emergency_signs"))
            disclaimer = _safe_text(
                final_result.get("disclaimer"),
                "本结果仅供参考，不替代线下面诊。",
            )

            lines = [
                "【初步判断】",
                diagnosis,
                "",
                "【复诊建议】",
                "\n".join(f"- {item}" for item in follow_up) if follow_up else "- 请结合线下面诊。",
                "",
                "【立即就医信号】",
                "\n".join(f"- {item}" for item in emergency) if emergency else "- 症状明显加重请急诊。",
                "",
                "【免责声明】",
                disclaimer,
            ]
            return "\n".join(lines).strip()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        update = self.prepare(state)

        merged_state = dict(state)
        merged_state.update(update)

        assistant_reply = self.fallback_reply(merged_state)
        history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
        history.append({"role": "assistant", "content": assistant_reply})

        update["assistant_reply"] = assistant_reply
        update["history"] = history
        return update

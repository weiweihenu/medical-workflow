# app/state.py
from typing import Any, Dict, List, Literal, Optional, TypedDict

# 路由科室
Department = Literal["呼吸科", "心血管科", "消化内科", "全科"]

# 性别与流程动作
SexType = Literal["male", "female", "other"]
NextAction = Literal["ask_user_more", "continue", "done"]


class ChatTurn(TypedDict):
    """一条对话消息。"""
    role: Literal["user", "assistant", "system"]
    content: str


class PatientInfo(TypedDict, total=False):
    """询问智能体抽取后的结构化病人信息。"""
    age: Optional[int]
    sex: Optional[SexType]

    chief_complaint: str
    duration: str
    severity: str
    symptoms: List[str]

    allergies: List[str]
    chronic_diseases: List[str]
    current_meds: List[str]

    additional_notes: str


class RouteDecision(TypedDict, total=False):
    """路由智能体输出。"""
    department: Department
    reason: str


class MedicationSuggestion(TypedDict, total=False):
    """专科给出的可讨论用药建议（非处方）。"""
    name: str
    purpose: str
    otc: bool


class SpecialistResult(TypedDict, total=False):
    """专科智能体输出。"""
    preliminary_assessment: str
    possible_diagnoses: List[str]
    recommended_checks: List[str]
    medication_suggestions: List[MedicationSuggestion]
    risk_alerts: List[str]


class FinalResult(TypedDict, total=False):
    """总结智能体输出。"""
    diagnosis_summary: str
    prescription_advice: List[str]
    home_care: List[str]
    follow_up: List[str]
    emergency_signs: List[str]
    disclaimer: str


class MedState(TypedDict, total=False):
    """
    LangGraph 全局状态：
    各节点都读取/更新这个状态。
    """
    # 当前轮输入
    user_input: str

    # 对话历史
    history: List[ChatTurn]

    # 业务中间结果
    patient_info: PatientInfo
    route: RouteDecision
    specialist_result: SpecialistResult
    final_result: FinalResult

    # 当前轮给用户的回复
    assistant_reply: str

    # 流程控制字段
    next_action: NextAction

    # 上传文件提取后的文本，供路由/专科/总结使用
    documents_text: List[str]


def create_initial_state() -> MedState:
    """初始化 state，避免主程序里重复写。"""
    return {
        "history": [],
        "patient_info": {},
        "next_action": "continue",
    }

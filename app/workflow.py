from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents import IntakeAgent, RouterAgent, SpecialistAgent, SummaryAgent
from app.llm_client import LLMClient
from app.state import MedState


def _after_intake(state: MedState) -> Literal["ask_more", "to_router"]:
    """
    Intake 节点后的条件分支：
    - ask_user_more: 信息不足，结束本轮，等待用户继续补充
    - 其他情况: 进入 Router 节点
    """
    next_action = str(state.get("next_action", "")).strip()
    if next_action == "ask_user_more":
        return "ask_more"
    return "to_router"


def create_workflow(llm: LLMClient):
    """
    创建四智能体工作流：
    intake -> (END 或 router) -> specialist -> summary -> END
    """
    intake_agent = IntakeAgent(llm)
    router_agent = RouterAgent(llm)
    specialist_agent = SpecialistAgent(llm)
    summary_agent = SummaryAgent(llm)

    graph = StateGraph(MedState)

    # 1) 注册节点
    graph.add_node("intake", intake_agent)
    graph.add_node("router", router_agent)
    graph.add_node("specialist", specialist_agent)
    graph.add_node("summary", summary_agent)

    # 2) 设置入口
    graph.set_entry_point("intake")

    # 3) Intake 条件边
    graph.add_conditional_edges(
        "intake",
        _after_intake,
        {
            "ask_more": END,      # 先结束，等待用户补充信息
            "to_router": "router" # 信息足够，继续主流程
        },
    )

    # 4) 主干链路
    graph.add_edge("router", "specialist")
    graph.add_edge("specialist", "summary")
    graph.add_edge("summary", END)

    # 5) 编译工作流
    return graph.compile()


def create_initial_state() -> MedState:
    """
    初始化会话状态（建议后端创建 session 时调用）。
    注意：documents_text 用于承载上传检查单/纸质材料提取文本。
    """
    return {
        "history": [],
        "patient_info": {},
        "documents_text": [],
        "next_action": "continue",
    }

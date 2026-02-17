from typing import Dict, List, Optional


# 内部字段 -> 用户可读中文问题（带示例）
FIELD_QUESTION_MAP: Dict[str, str] = {
    "chief_complaint": "你现在最不舒服的主要症状是什么？例如：咳嗽、胸闷、发热。",
    "duration": "这些症状持续多久了？例如：3天、1周、1个月。",
    "severity": "症状严重程度如何？可用“轻/中/重”或“0-10分”（0=无症状，10=最严重）。例如：6分，夜里咳嗽会醒。",
    "age": "你的年龄是多少？例如：32岁。",
    "sex": "你的生理性别是？例如：男/女。",
    "symptoms": "还有哪些伴随症状？例如：流涕、咽痛、气短。",
    "allergies": "有药物或食物过敏吗？没有可写“无”。",
    "chronic_diseases": "有慢性病吗？例如：高血压、糖尿病；没有可写“无”。",
    "current_meds": "目前正在使用哪些药物？没有可写“无”。",
}


def normalize_missing_questions(
    missing_fields: Optional[List[str]],
    missing_questions: Optional[List[str]],
) -> List[str]:
    """
    把模型输出的问题统一成用户可读中文：
    - 如果模型只返回了 'severity' 这种键名，自动翻译成中文问题
    - 如果模型没给问题，则根据 missing_fields 自动生成
    """
    result: List[str] = []

    # 先处理模型给的 missing_questions
    for item in (missing_questions or []):
        text = str(item).strip()
        if not text:
            continue

        # 如果模型只给了字段名（如 severity），进行翻译
        if text in FIELD_QUESTION_MAP:
            text = FIELD_QUESTION_MAP[text]

        if text not in result:
            result.append(text)

    # 如果还没有问题，则根据 missing_fields 生成
    if not result:
        for field in (missing_fields or []):
            key = str(field).strip()
            if key in FIELD_QUESTION_MAP and FIELD_QUESTION_MAP[key] not in result:
                result.append(FIELD_QUESTION_MAP[key])

    return result

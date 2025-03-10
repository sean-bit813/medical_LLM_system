"""
意图检测模块 - 分析用户输入的意图
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union

from ..llm.api import generate_simple_response

# 配置日志
logger = logging.getLogger(__name__)

# 定义意图类型
INTENT_TYPES = {
    "report_symptom": "报告症状",
    "ask_question": "咨询问题",
    "request_info": "请求信息",
    "express_concern": "表达担忧",
    "share_history": "分享病史",
    "request_advice": "请求建议",
    "emergency": "紧急求助",
    "greeting": "问候",
    "farewell": "道别",
    "gratitude": "表达感谢",
    "clarification": "请求澄清",
    "confirmation": "确认信息",
    "rejection": "拒绝建议",
    "follow_up": "随访",
    "other": "其他"
}


def detect_intent(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    检测用户输入的意图

    Args:
        text: 用户输入文本
        context: 可选的对话上下文信息

    Returns:
        包含意图分析结果的字典，例如：
        {
            "primary_intent": "report_symptom",
            "confidence": 0.85,
            "secondary_intents": [
                {"intent": "request_advice", "confidence": 0.45}
            ],
            "entities": {
                "symptoms": ["头痛", "发热"]
            }
        }
    """
    # 简单问候语直接处理，避免调用LLM
    simple_greetings = ["你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "开始", "start"]
    if text.strip().lower() in simple_greetings:
        return {
            "primary_intent": "greeting",
            "confidence": 0.98,
            "secondary_intents": [],
            "entities": {}
        }

    # 简单告别语也直接处理
    simple_farewells = ["再见", "拜拜", "谢谢", "谢谢你", "goodbye", "bye", "thanks", "thank you"]
    if text.strip().lower() in simple_farewells:
        intent = "gratitude" if "谢" in text or "thank" in text.lower() else "farewell"
        return {
            "primary_intent": intent,
            "confidence": 0.98,
            "secondary_intents": [],
            "entities": {}
        }

    # 构建意图分析的上下文
    context_str = ""
    if context:
        context_str = "对话上下文:\n"
        # 添加当前医疗信息
        if "medical_info" in context:
            for key, value in context["medical_info"].items():
                context_str += f"- {key}: {value}\n"
        # 添加最近对话
        if "dialogue" in context and isinstance(context["dialogue"], list):
            recent_dialogue = context["dialogue"][-3:] if len(context["dialogue"]) > 3 else context["dialogue"]
            context_str += "最近对话:\n"
            for turn in recent_dialogue:
                context_str += f"- {turn['role']}: {turn['content']}\n"

    # 构造系统提示词
    system_prompt = f"""你是一个专业的医疗对话意图分析助手。请分析以下用户输入的主要意图和可能的次要意图。

    可能的意图类型包括：
    {', '.join([f"{k}({v})" for k, v in INTENT_TYPES.items()])}

    请分析用户输入，判断主要意图和可能的次要意图，并提取相关实体（如症状、药物等）。

    请以JSON格式返回分析结果，包含以下字段：
    - primary_intent: 主要意图
    - confidence: 置信度（0-1的浮点数）
    - secondary_intents: 次要意图列表，每个意图包含intent和confidence
    - entities: 输入中提到的实体，按类型分组

    不需要解释理由，只需返回JSON结果。
    """

    # 准备提示词
    prompt = f"{context_str}\n\n用户输入: {text}"

    try:
        # 调用LLM API
        response = generate_simple_response(prompt, system_prompt)

        # 尝试解析JSON
        try:
            result = json.loads(response)
            logger.debug(f"意图检测结果: {result}")
            return result
        except json.JSONDecodeError:
            logger.error(f"意图检测JSON解析失败: {response}")
            # 如果无法解析，返回通用意图
            return {
                "primary_intent": "other",
                "confidence": 0.5,
                "secondary_intents": [],
                "entities": {}
            }

    except Exception as e:
        logger.error(f"意图检测出错: {str(e)}")
        return {
            "primary_intent": "other",
            "confidence": 0.3,
            "secondary_intents": [],
            "entities": {},
            "error": str(e)
        }


def is_emergency_intent(text: str) -> Dict[str, Any]:
    """
    专门检测是否表达紧急情况的意图

    Args:
        text: 用户输入文本

    Returns:
        包含紧急性分析结果的字典
    """
    # 构造系统提示词
    system_prompt = """你是一个专业的医疗紧急情况检测助手。请评估用户输入是否描述了需要紧急医疗干预的情况。

    紧急医疗情况包括但不限于：
    - 剧烈胸痛或胸部压迫感
    - 严重呼吸困难
    - 意识不清或意识改变
    - 大量出血
    - 严重过敏反应（如喉咙肿胀、呼吸急促）
    - 严重头痛伴随视力变化或意识改变
    - 严重烧伤或外伤
    - 癫痫发作或持续抽搐

    请以JSON格式返回分析结果，包含以下字段：
    - is_emergency: 布尔值，表示是否紧急情况
    - confidence: 置信度（0-1的浮点数）
    - reason: 判断理由
    - severity: 严重程度评分（1-10）

    不需要解释理由，只需返回JSON结果。
    """

    try:
        # 调用LLM API
        response = generate_simple_response(text, system_prompt)

        # 尝试解析JSON
        try:
            result = json.loads(response)
            logger.debug(f"紧急意图检测结果: {result}")
            return result
        except json.JSONDecodeError:
            logger.error(f"紧急意图检测JSON解析失败: {response}")
            # 如果无法解析，假设非紧急情况
            return {
                "is_emergency": False,
                "confidence": 0.5,
                "reason": "无法解析结果",
                "severity": 3
            }

    except Exception as e:
        logger.error(f"紧急意图检测出错: {str(e)}")
        return {
            "is_emergency": False,
            "confidence": 0.3,
            "reason": f"检测出错: {str(e)}",
            "severity": 3,
            "error": str(e)
        }
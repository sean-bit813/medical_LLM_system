# src/personalization/preference_detector.py
"""
偏好检测模块 - 从用户消息中检测交互偏好
"""
import json
import logging
from typing import Dict, Any, List

from ..llm.api import generate_simple_response

# 配置日志
logger = logging.getLogger(__name__)


class PreferenceDetector:
    """用户偏好检测器，分析用户消息中的偏好信号"""

    def __init__(self):
        """初始化偏好检测器"""
        logger.info("偏好检测器初始化完成")

    def detect_preferences(self, message: str, dialogue_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从用户消息中检测偏好

        Args:
            message: 用户消息
            dialogue_history: 对话历史

        Returns:
            检测到的偏好信息，如沟通风格和详细程度
        """
        # 准备最近的对话历史
        recent_history = dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history

        # 构建上下文描述
        context_description = "对话历史:\n"
        for turn in recent_history:
            role = "医生" if turn.get("role") == "doctor" else "患者"
            context_description += f"{role}: {turn.get('content', '')}\n"

        # 构造系统提示词
        system_prompt = """你是一个专业的医疗交互偏好分析助手。请从用户的消息中分析其偏好信号。

        请重点关注以下方面:
        1. 沟通风格偏好（专业/友好/中性）
        2. 信息详细程度偏好（简单/正常/详细）
        3. 情感倾向（急躁/焦虑/冷静/困惑等）
        4. 专业术语使用偏好（是否使用或理解医学术语）

        请以JSON格式返回分析结果，包含以下字段:
        - communication_style: "professional" / "friendly" / "neutral"
        - detail_level: "simple" / "normal" / "detailed"
        - emotion: 情感倾向描述
        - terminology_preference: "technical" / "layman" / "mixed"
        - confidence: 0-1的浮点数，表示分析的置信度
        """

        # 准备提示词
        prompt = f"{context_description}\n\n当前消息: {message}"

        try:
            # 调用LLM API
            response = generate_simple_response(prompt, system_prompt)

            # 尝试解析结果
            try:
                result = json.loads(response)
                logger.debug(f"偏好分析结果: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"偏好分析JSON解析失败: {response}")
                return {
                    "communication_style": "neutral",
                    "detail_level": "normal",
                    "emotion": "neutral",
                    "terminology_preference": "mixed",
                    "confidence": 0.5
                }

        except Exception as e:
            logger.error(f"偏好分析出错: {str(e)}")
            return {
                "communication_style": "neutral",
                "detail_level": "normal",
                "emotion": "neutral",
                "terminology_preference": "mixed",
                "confidence": 0.5,
                "error": str(e)
            }

    def detect_communication_style(self, message: str) -> Dict[str, Any]:
        """分析用户沟通风格偏好

        Args:
            message: 用户消息

        Returns:
            沟通风格分析结果
        """
        # 构造系统提示词
        system_prompt = """你是一个专业的交流风格分析助手。请分析用户的消息，确定其偏好的沟通风格。

        沟通风格可以分为以下几类:
        - professional: 偏好专业化、直接的沟通方式，使用医学术语
        - friendly: 偏好温暖、友好的沟通方式，使用日常语言
        - neutral: 介于专业和友好之间的平衡沟通方式

        请以JSON格式返回分析结果，包含以下字段:
        - style: "professional" / "friendly" / "neutral"
        - confidence: 0-1的浮点数，表示分析的置信度
        - reasoning: 简要分析理由
        """

        try:
            # 调用LLM API
            response = generate_simple_response(message, system_prompt)

            # 尝试解析结果
            try:
                result = json.loads(response)
                logger.debug(f"沟通风格分析结果: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"沟通风格分析JSON解析失败: {response}")
                return {
                    "style": "neutral",
                    "confidence": 0.5,
                    "reasoning": "无法解析结果"
                }

        except Exception as e:
            logger.error(f"沟通风格分析出错: {str(e)}")
            return {
                "style": "neutral",
                "confidence": 0.5,
                "reasoning": f"分析出错: {str(e)}"
            }

    def detect_detail_level(self, message: str) -> Dict[str, Any]:
        """分析用户对信息详细程度的偏好

        Args:
            message: 用户消息

        Returns:
            详细程度偏好分析结果
        """
        # 构造系统提示词
        system_prompt = """你是一个专业的信息偏好分析助手。请分析用户的消息，确定其偏好的信息详细程度。

        详细程度可以分为以下几类:
        - simple: 偏好简洁的信息，只需要基本要点
        - normal: 偏好适度详细的信息，包含一些解释
        - detailed: 偏好非常详细的信息，包含全面的解释和背景

        请以JSON格式返回分析结果，包含以下字段:
        - detail_level: "simple" / "normal" / "detailed"
        - confidence: 0-1的浮点数，表示分析的置信度
        - reasoning: 简要分析理由
        """

        try:
            # 调用LLM API
            response = generate_simple_response(message, system_prompt)

            # 尝试解析结果
            try:
                result = json.loads(response)
                logger.debug(f"详细程度偏好分析结果: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"详细程度偏好分析JSON解析失败: {response}")
                return {
                    "detail_level": "normal",
                    "confidence": 0.5,
                    "reasoning": "无法解析结果"
                }

        except Exception as e:
            logger.error(f"详细程度偏好分析出错: {str(e)}")
            return {
                "detail_level": "normal",
                "confidence": 0.5,
                "reasoning": f"分析出错: {str(e)}"
            }
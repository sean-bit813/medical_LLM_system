"""
上下文分析器 - 分析对话上下文，提供上下文感知能力
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..llm.api import generate_simple_response

# 配置日志
logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """上下文分析器，提供对话上下文分析能力"""

    def __init__(self, memory_manager=None):
        """
        初始化上下文分析器

        Args:
            memory_manager: 记忆管理器，用于访问短期、中期和长期记忆
        """
        self.memory_manager = memory_manager
        logger.info("上下文分析器初始化完成")

    def analyze_context(self, current_message: str, dialogue_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析当前消息在对话上下文中的含义

        Args:
            current_message: 当前用户消息
            dialogue_context: 对话上下文，包含历史对话、已收集信息等

        Returns:
            上下文分析结果
        """
        # 提取对话历史
        dialogue_history = dialogue_context.get("dialogue", [])
        recent_history = dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history

        # 提取医疗信息
        medical_info = dialogue_context.get("medical_info", {})

        # 构建上下文描述
        context_description = "对话历史:\n"
        for turn in recent_history:
            role = "医生" if turn.get("role") == "doctor" else "患者"
            context_description += f"{role}: {turn.get('content', '')}\n"

        context_description += "\n已收集的医疗信息:\n"
        for key, value in medical_info.items():
            context_description += f"- {key}: {value}\n"

        # 构造系统提示词
        system_prompt = """你是一个专业的医疗对话上下文分析助手。请分析当前用户消息在给定对话上下文中的含义和相关性。
        重点关注:
        1. 消息是否引用了之前提到的症状、药物或其他医疗信息
        2. 消息是否提供了新的医疗信息
        3. 消息是否修改或纠正了之前的信息
        4. 消息的情感倾向（如担忧、困惑、满意等）

        请以JSON格式返回分析结果，包含以下字段:
        - references: 引用的先前信息列表
        - new_info: 新提供的信息
        - corrections: 对先前信息的修改
        - emotion: 情感倾向
        - relevance: 与当前主题的相关性(0-1)
        """

        # 准备提示词
        prompt = f"{context_description}\n\n当前消息: {current_message}"

        try:
            # 调用LLM API
            response = generate_simple_response(prompt, system_prompt)

            # 尝试解析结果
            try:
                result = json.loads(response)
                logger.debug(f"上下文分析结果: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"上下文分析JSON解析失败: {response}")
                return {
                    "references": [],
                    "new_info": {},
                    "corrections": {},
                    "emotion": "neutral",
                    "relevance": 0.5
                }

        except Exception as e:
            logger.error(f"上下文分析出错: {str(e)}")
            return {
                "references": [],
                "new_info": {},
                "corrections": {},
                "emotion": "neutral",
                "relevance": 0.5,
                "error": str(e)
            }

    def cross_reference_symptoms(self, symptoms: List[Dict[str, Any]], medical_context: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """
        对症状进行交叉引用，与已知病史和症状比对

        Args:
            symptoms: 症状列表
            medical_context: 医疗上下文，包含病史、过去症状等

        Returns:
            增强后的症状列表，添加了与历史的关联信息
        """
        if not symptoms:
            return []

        # 提取病史和过去症状
        past_symptoms = medical_context.get("past_symptoms", [])
        medical_history = medical_context.get("medical_history", "")

        # 如果没有过去症状和病史，直接返回原始症状列表
        if not past_symptoms and not medical_history:
            return symptoms

        # 构建上下文描述
        context_description = "患者病史:\n"
        context_description += medical_history + "\n\n"

        context_description += "过去症状:\n"
        for symptom in past_symptoms:
            if isinstance(symptom, dict) and "name" in symptom:
                context_description += f"- {symptom['name']}"
                if "duration" in symptom:
                    context_description += f" (持续: {symptom['duration']})"
                context_description += "\n"
            elif isinstance(symptom, str):
                context_description += f"- {symptom}\n"

        context_description += "\n当前症状:\n"
        for symptom in symptoms:
            if isinstance(symptom, dict) and "name" in symptom:
                context_description += f"- {symptom['name']}"
                if "duration" in symptom:
                    context_description += f" (持续: {symptom['duration']})"
                context_description += "\n"
            elif isinstance(symptom, str):
                context_description += f"- {symptom}\n"

        # 构造系统提示词
        system_prompt = """你是一个专业的医疗症状分析助手。请分析当前症状与患者病史和过去症状的关系。

        对于每个当前症状，请确定:
        1. 这是新出现的症状，还是之前就有的症状
        2. 症状是否与已知病史相关
        3. 症状是否有加重或改善
        4. 是否需要特别关注的症状

        请为每个症状添加以下字段:
        - is_new: 布尔值，表示是否为新症状
        - related_to_history: 布尔值，表示是否与病史相关
        - changes: 变化情况 ("improved", "worsened", "unchanged", "unknown")
        - attention_needed: 布尔值，表示是否需要特别关注
        - notes: 相关说明

        请以JSON格式返回分析结果，返回一个包含所有症状分析的数组。
        """

        try:
            # 调用LLM API
            response = generate_simple_response(context_description, system_prompt)

            # 尝试解析结果
            try:
                results = json.loads(response)
                logger.debug(f"症状交叉引用结果: {results}")

                # 确保结果是一个列表
                if not isinstance(results, list):
                    logger.warning(f"症状交叉引用结果不是列表: {results}")
                    return symptoms

                # 将分析结果合并到原始症状中
                enriched_symptoms = []

                # 确保结果和症状列表长度一致
                if len(results) == len(symptoms):
                    for i, symptom in enumerate(symptoms):
                        if isinstance(symptom, dict):
                            enriched_symptom = symptom.copy()
                            enriched_symptom.update(results[i])
                            enriched_symptoms.append(enriched_symptom)
                        else:
                            # 如果症状是字符串，创建一个字典
                            enriched_symptom = {
                                "name": symptom,
                                **results[i]
                            }
                            enriched_symptoms.append(enriched_symptom)
                else:
                    # 如果长度不一致，尝试按名称匹配
                    for symptom in symptoms:
                        symptom_name = symptom.get("name", symptom) if isinstance(symptom, dict) else symptom

                        # 查找匹配的分析结果
                        matched_result = None
                        for result in results:
                            result_name = result.get("name", "")
                            if result_name == symptom_name:
                                matched_result = result
                                break

                        if matched_result and isinstance(symptom, dict):
                            enriched_symptom = symptom.copy()
                            enriched_symptom.update(matched_result)
                            enriched_symptoms.append(enriched_symptom)
                        elif matched_result:
                            enriched_symptom = {
                                "name": symptom,
                                **matched_result
                            }
                            enriched_symptoms.append(enriched_symptom)
                        else:
                            # 如果没有匹配的结果，使用原始症状
                            enriched_symptoms.append(symptom)

                return enriched_symptoms

            except json.JSONDecodeError:
                logger.error(f"症状交叉引用JSON解析失败: {response}")
                return symptoms

        except Exception as e:
            logger.error(f"症状交叉引用出错: {str(e)}")
            return symptoms

    def detect_contradiction(self, message: str, medical_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测用户消息是否与已收集的医疗信息存在矛盾

        Args:
            message: 用户消息
            medical_info: 已收集的医疗信息

        Returns:
            矛盾检测结果
        """
        if not medical_info:
            return {"has_contradiction": False, "contradictions": {}}

        # 构建医疗信息描述
        info_description = "已收集的医疗信息:\n"
        for key, value in medical_info.items():
            info_description += f"- {key}: {value}\n"

        # 构造系统提示词
        system_prompt = """你是一个专业的医疗信息一致性检查助手。请检查用户消息是否与已收集的医疗信息存在矛盾。

        请特别关注:
        1. 时间信息的矛盾（如症状持续时间不一致）
        2. 症状描述的矛盾（如严重程度、性质的前后不一致）
        3. 个人信息的矛盾（如年龄、性别的前后不一致）
        4. 医疗史的矛盾（如病史、用药史的前后不一致）

        请以JSON格式返回分析结果，包含以下字段:
        - has_contradiction: 布尔值，表示是否存在矛盾
        - contradictions: 对象，键为信息字段，值为描述矛盾的对象
          - original: 原始信息
          - new: 新信息
          - description: 矛盾描述
        """

        # 准备提示词
        prompt = f"{info_description}\n\n用户消息: {message}"

        try:
            # 调用LLM API
            response = generate_simple_response(prompt, system_prompt)

            # 尝试解析结果
            try:
                result = json.loads(response)
                logger.debug(f"矛盾检测结果: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"矛盾检测JSON解析失败: {response}")
                return {"has_contradiction": False, "contradictions": {}}

        except Exception as e:
            logger.error(f"矛盾检测出错: {str(e)}")
            return {
                "has_contradiction": False,
                "contradictions": {},
                "error": str(e)
            }
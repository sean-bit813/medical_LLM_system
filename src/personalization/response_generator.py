# src/personalization/response_generator.py
"""
个性化响应生成模块 - 根据用户偏好生成定制化回复
"""
import logging
from typing import Dict, Any, Optional

from ..llm.api import generate_simple_response
from .user_profile import UserProfile

# 配置日志
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """个性化响应生成器，根据用户偏好生成定制化回复"""

    def __init__(self):
        """初始化响应生成器"""
        logger.info("个性化响应生成器初始化完成")

    def generate_response(self, message: str, profile: UserProfile, template: str,
                          additional_info: Dict[str, Any] = None) -> str:
        """生成个性化响应

        Args:
            message: 用户消息
            profile: 用户画像
            template: 响应模板
            additional_info: 额外信息字典

        Returns:
            生成的个性化响应
        """
        # 获取用户偏好
        communication_style = profile.get_communication_style()
        detail_level = profile.get_detail_level()

        # 获取基本信息和医疗历史
        user_name = profile.basic_info.get('name', '')
        age = profile.basic_info.get('age', '')
        gender = profile.basic_info.get('gender', '')
        key_medical_history = self._format_medical_history(profile.medical_history)

        # 构建个性化提示词
        personalized_prompt = f"""
        用户信息：
        - 名字：{user_name}
        - 年龄：{age}
        - 性别：{gender}
        - 沟通风格偏好：{communication_style}
        - 信息详细程度偏好：{detail_level}

        医疗历史：
        {key_medical_history}

        用户消息：{message}

        响应模板：{template}
        """

        # 添加额外信息
        if additional_info:
            personalized_prompt += "\n额外信息：\n"
            for key, value in additional_info.items():
                personalized_prompt += f"- {key}: {value}\n"

        # 构造系统提示词
        system_prompt = f"""你是一个专业的医疗助手，需要根据用户偏好生成个性化响应。

        用户偏好通信风格：{communication_style}
        - professional: 使用专业、直接的语言，可以使用适当的医学术语
        - friendly: 使用温暖、亲切的语言，避免过多医学术语，注重共情
        - neutral: 使用平衡的语言，根据上下文适当调整专业程度

        用户偏好信息详细程度：{detail_level}
        - simple: 提供简洁的信息，只包含必要要点
        - normal: 提供适度详细的信息，包含一些解释
        - detailed: 提供全面详细的信息，包含充分的解释和背景

        请严格根据这些偏好和提供的响应模板生成个性化回复。
        """

        try:
            # 调用LLM API
            response = generate_simple_response(personalized_prompt, system_prompt)
            logger.debug(f"生成个性化响应: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"生成个性化响应出错: {str(e)}")
            # 返回原始模板作为后备
            return template

    def _format_medical_history(self, medical_history: Dict[str, Any]) -> str:
        """格式化医疗历史信息

        Args:
            medical_history: 医疗历史字典

        Returns:
            格式化的医疗历史文本
        """
        if not medical_history:
            return "无显著病史"

        formatted = []

        # 处理疾病史
        if 'diseases' in medical_history:
            diseases = medical_history['diseases']
            if isinstance(diseases, list):
                formatted.append("疾病史: " + ", ".join(diseases))
            elif isinstance(diseases, str):
                formatted.append(f"疾病史: {diseases}")

        # 处理过敏史
        if 'allergies' in medical_history:
            allergies = medical_history['allergies']
            if isinstance(allergies, list):
                formatted.append("过敏史: " + ", ".join(allergies))
            elif isinstance(allergies, str):
                formatted.append(f"过敏史: {allergies}")

        # 处理手术史
        if 'surgeries' in medical_history:
            surgeries = medical_history['surgeries']
            if isinstance(surgeries, list):
                formatted.append("手术史: " + ", ".join(surgeries))
            elif isinstance(surgeries, str):
                formatted.append(f"手术史: {surgeries}")

        # 处理其他项
        for key, value in medical_history.items():
            if key not in ['diseases', 'allergies', 'surgeries']:
                if isinstance(value, list):
                    formatted.append(f"{key}: " + ", ".join(value))
                elif isinstance(value, str):
                    formatted.append(f"{key}: {value}")

        return "\n".join(formatted)

    def adapt_response_style(self, response: str, profile: UserProfile) -> str:
        """根据用户偏好调整响应风格

        Args:
            response: 原始响应
            profile: 用户画像

        Returns:
            调整风格后的响应
        """
        # 获取用户偏好
        communication_style = profile.get_communication_style()
        detail_level = profile.get_detail_level()

        # 构造系统提示词
        system_prompt = f"""你是一个专业的医疗文本风格调整助手。请根据以下偏好调整医疗响应的风格：

        通信风格：{communication_style}
        - professional: 使用专业、直接的语言，可以使用适当的医学术语
        - friendly: 使用温暖、亲切的语言，避免过多医学术语，注重共情
        - neutral: 使用平衡的语言，根据上下文适当调整专业程度

        信息详细程度：{detail_level}
        - simple: 提供简洁的信息，只包含必要要点，整体字数减少约30%
        - normal: 保持原有详细程度
        - detailed: 提供更详细的解释，可适当增加内容

        请调整以下医疗响应的风格和详细程度，保持原有的医疗建议不变。

        原始响应：
        {response}
        """

        # 直接返回原始响应的情况
        if communication_style == "neutral" and detail_level == "normal":
            return response

        try:
            # 调用LLM API
            adjusted_response = generate_simple_response(system_prompt, temperature=0.3, max_tokens=len(response) * 2)
            logger.debug(f"调整响应风格: {adjusted_response[:100]}...")
            return adjusted_response

        except Exception as e:
            logger.error(f"调整响应风格出错: {str(e)}")
            # 返回原始响应作为后备
            return response

    def personalize_greeting(self, profile: UserProfile) -> str:
        """生成个性化问候

        Args:
            profile: 用户画像

        Returns:
            个性化问候语
        """
        # 获取用户名和沟通风格
        user_name = profile.basic_info.get('name', '')
        communication_style = profile.get_communication_style()

        if not user_name:
            # 无用户名时的通用问候
            if communication_style == "professional":
                return "您好，我是您的医疗助手。有什么可以帮助您的吗？"
            elif communication_style == "friendly":
                return "你好呀！我是你的医疗小助手，有什么我能帮上忙的吗？"
            else:  # neutral
                return "您好，我是您的医疗助手。请问有什么可以帮您？"
        else:
            # 有用户名时的个性化问候
            if communication_style == "professional":
                return f"{user_name}您好，我是您的医疗助手。有什么可以帮助您的吗？"
            elif communication_style == "friendly":
                return f"你好，{user_name}！很高兴再次见到你。今天有什么我能帮你的吗？"
            else:  # neutral
                return f"{user_name}您好，我是您的医疗助手。请问有什么可以帮您？"

    def add_personalized_parts(self, base_response: str, profile: UserProfile,
                               medical_info: Dict[str, Any]) -> str:
        """添加个性化信息部分

        Args:
            base_response: 基础响应
            profile: 用户画像
            medical_info: 医疗信息

        Returns:
            添加个性化部分后的完整响应
        """
        personalized_parts = []

        # 1. 基于病史的提醒
        if 'allergies' in profile.medical_history:
            allergies = profile.medical_history['allergies']
            if isinstance(allergies, list) and allergies:
                allergy_list = ", ".join(allergies)
                personalized_parts.append(f"根据您的过敏史（{allergy_list}），请务必在用药前告知医生。")
            elif isinstance(allergies, str) and allergies:
                personalized_parts.append(f"根据您的过敏史（{allergies}），请务必在用药前告知医生。")

        # 2. 症状进展追踪
        for symptom_name, symptom_info in profile.symptom_entities.items():
            if 'first_mentioned' in symptom_info and 'severity' in symptom_info:
                # 检查当前医疗信息中是否有该症状的最新严重程度
                current_severity = None
                if 'symptom_severities' in medical_info:
                    current_severity = medical_info['symptom_severities'].get(symptom_name)

                prev_severity = symptom_info.get('severity')

                if current_severity and prev_severity and current_severity != prev_severity:
                    if current_severity > prev_severity:
                        personalized_parts.append(f"注意到您的{symptom_name}症状有所加重，请密切关注。")
                    else:
                        personalized_parts.append(f"您的{symptom_name}症状相比之前有所改善，这是个好现象。")

        # 3. 根据用户偏好添加信息详细程度调整
        detail_level = profile.get_detail_level()
        if detail_level == "detailed" and len(base_response) < 1000:
            # 为偏好详细信息的用户添加扩展解释
            system_prompt = """你是一个专业的医疗信息扩展助手。请对下面的医疗建议添加更详细的解释，包括：
            1. 医学术语的详细解释
            2. 原理说明
            3. 可能的替代方案
            4. 扩展的预防建议

            原始建议：
            """

            try:
                extended_parts = generate_simple_response(system_prompt + base_response)
                if len(extended_parts) > len(base_response) * 1.2:  # 确保扩展有意义
                    personalized_parts.append("\n扩展信息：\n" + extended_parts)
            except Exception as e:
                logger.error(f"生成扩展信息出错: {str(e)}")

        # 组合最终响应
        if personalized_parts:
            return base_response + "\n\n" + "\n".join(personalized_parts)
        return base_response
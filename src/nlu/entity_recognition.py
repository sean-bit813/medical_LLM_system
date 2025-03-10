"""
实体识别模块 - 从自然语言文本中识别医疗相关实体
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union

from ..llm.api import generate_simple_response

# 配置日志
logger = logging.getLogger(__name__)


def symptom_entity_recognition(text: str) -> Dict[str, Any]:
    """
    使用大模型识别文本中的症状实体

    Args:
        text: 待分析的文本

    Returns:
        包含识别结果的字典，例如：
        {
            "symptoms": ["头痛", "咳嗽", "发热"],
            "context": "原始文本内容"
        }
    """
    try:
        # 构造系统提示词
        system_prompt = """你是一个专业的医疗信息处理助手。请从用户输入的文本中识别并提取所有提到的症状实体。
        要求：
        1. 只返回JSON格式结果，仅包含symptoms数组
        2. 症状实体必须在用户输入文本中出现
        3. 排除非症状描述（如药品、检查项目等）
        4. 如果描述了症状的特性（如位置、程度、持续时间等），将其一并提取"""

        # 调用LLM API
        response = generate_simple_response(text, system_prompt)

        # 尝试解析JSON
        try:
            result = json.loads(response)
            # 确保返回格式一致
            if "symptoms" not in result:
                result["symptoms"] = []
            result["context"] = text
            return result
        except json.JSONDecodeError:
            logger.error(f"症状实体识别JSON解析失败: {response}")
            # 尝试从非标准格式中提取信息
            symptoms = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line[0] != '{' and line[-1] != '}':
                    symptoms.append(line.strip('"').strip("'").strip())
            return {"symptoms": symptoms, "context": text}

    except Exception as e:
        logger.error(f"症状实体识别出错: {str(e)}")
        return {"symptoms": [], "context": text, "error": str(e)}


def medication_entity_recognition(text: str) -> Dict[str, Any]:
    """
    识别文本中的药物实体

    Args:
        text: 待分析的文本

    Returns:
        包含识别结果的字典
    """
    try:
        # 构造系统提示词
        system_prompt = """你是一个专业的医疗信息处理助手。请从用户输入的文本中识别并提取所有提到的药物实体。
        要求：
        1. 只返回JSON格式结果，包含medications数组
        2. 药物实体必须在用户输入文本中出现
        3. 如果描述了药物的剂量、使用方法等信息，将其一并提取
        4. 区分处方药和非处方药"""

        # 调用LLM API
        response = generate_simple_response(text, system_prompt)

        # 尝试解析JSON
        try:
            result = json.loads(response)
            # 确保返回格式一致
            if "medications" not in result:
                result["medications"] = []
            result["context"] = text
            return result
        except json.JSONDecodeError:
            logger.error(f"药物实体识别JSON解析失败: {response}")
            return {"medications": [], "context": text}

    except Exception as e:
        logger.error(f"药物实体识别出错: {str(e)}")
        return {"medications": [], "context": text, "error": str(e)}


def medical_entity_recognition(text: str, entity_types: List[str] = None) -> Dict[str, Any]:
    """
    综合识别文本中的医疗实体

    Args:
        text: 待分析的文本
        entity_types: 要识别的实体类型列表，默认为["symptoms", "medications", "diseases", "tests"]

    Returns:
        包含识别结果的字典
    """
    if entity_types is None:
        entity_types = ["symptoms", "medications", "diseases", "tests"]

    entity_types_str = ", ".join(entity_types)

    try:
        # 构造系统提示词
        system_prompt = f"""你是一个专业的医疗信息处理助手。请从用户输入的文本中识别并提取以下医疗实体：{entity_types_str}。
        要求：
        1. 只返回JSON格式结果
        2. 实体必须在用户输入文本中出现
        3. 对于每种实体类型，提取出现的所有实例
        4. 如果实体有其他属性（如严重程度、持续时间、用量等），将其一并提取"""

        # 调用LLM API
        response = generate_simple_response(text, system_prompt)

        # 尝试解析JSON
        try:
            result = json.loads(response)
            # 确保所有请求的实体类型都存在
            for entity_type in entity_types:
                if entity_type not in result:
                    result[entity_type] = []
            result["context"] = text
            return result
        except json.JSONDecodeError:
            logger.error(f"医疗实体识别JSON解析失败: {response}")
            return {entity_type: [] for entity_type in entity_types}

    except Exception as e:
        logger.error(f"医疗实体识别出错: {str(e)}")
        result = {entity_type: [] for entity_type in entity_types}
        result["context"] = text
        result["error"] = str(e)
        return result
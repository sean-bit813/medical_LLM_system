# src/llm/api.py
from typing import List, Dict, Optional
import logging
from openai import OpenAI
from ..app_config import LLM_CONFIG
from ..dialogue.states import DialogueState
from ..prompts.medical_prompts import SYSTEM_PROMPT, MEDICAL_PROMPTS

# 设置日志
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"]
)


def generate_response(context) -> str:
    """生成基于上下文和知识库的回复"""
    state_prompt_mapping = {
        DialogueState.DIAGNOSIS.value: 'diagnosis_template',
        DialogueState.MEDICAL_ADVICE.value: 'medical_advice_template',
        DialogueState.REFERRAL.value: 'referral_template',
        DialogueState.EDUCATION.value: 'education_template'
    }

    if context.state.value not in state_prompt_mapping:
        return "抱歉,我现在无法处理这个请求。"

    template_key = state_prompt_mapping[context.state.value]
    template = MEDICAL_PROMPTS[template_key]

    knowledge_context = context.medical_info.get('relevant_knowledge', '')

    prompt = template.format(
        all_info=context.medical_info.get('formatted_info', ''),
        diagnosis=context.medical_info.get("diagnosis", "未知"),
        urgency=context.medical_info.get('referral_urgency', 'non_urgent')
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"相关医学知识:\n{knowledge_context}\n\n用户信息:{prompt}"}  # knowledge_base
    ]

    logger.info(f"生成回复: 状态={context.state.value}, 模板={template_key}")

    try:
        completion = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API调用错误: {e}")
        return "抱歉,系统暂时无法生成回复。"


def generate_simple_response(prompt: str, system_prompt: Optional[str] = None, temperature: float = None,
                             max_tokens: int = None) -> str:
    """
    简化版的LLM调用，用于动态对话流程中的小型任务

    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词，默认为简单的医疗助手提示
        temperature: 温度参数
        max_tokens: 最大生成token数

    Returns:
        LLM生成的回复文本
    """
    if system_prompt is None:
        system_prompt = "你是一个专业的医疗助手，需要简洁明了地回答问题。"

    if temperature is None:
        temperature = LLM_CONFIG.get("temperature", 0.1)

    if max_tokens is None:
        max_tokens = LLM_CONFIG.get("max_tokens", 200)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    logger.info(f"简单LLM调用: temperature={temperature}, max_tokens={max_tokens}")

    try:
        completion = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"简单LLM API调用错误: {e}")
        return "无法获取回复，请重试。"
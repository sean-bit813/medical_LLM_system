# src/llm/api.py
from typing import List, Dict
from openai import OpenAI
from ..config import LLM_CONFIG
from ..dialogue.states import DialogueState
from ..prompts.medical_prompts import SYSTEM_PROMPT, MEDICAL_PROMPTS

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
        {"role": "user", "content": f"相关医学知识:\n{knowledge_context}\n\n用户信息:{prompt}"} # knowledge_base
    ]

    try:
        completion = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM API Error: {e}")
        return "抱歉,系统暂时无法生成回复。"
# src/dialogue/field_mappings.py
"""
医疗对话中使用的字段映射定义文件
提供详细的字段说明、中文映射和示例
"""
from ..config.loader import ConfigLoader

# 加载字段映射配置
mappings_config = ConfigLoader.load_json_config('field_mappings.json')

# 加载各类字段映射
COMBINED_INFO_MAPPING = mappings_config.get('combined_info_mapping', {})
BASE_INFO_MAPPING = mappings_config.get('base_info_mapping', {})
SYMPTOM_MAPPING = mappings_config.get('symptom_mapping', {})
LIFESTYLE_MAPPING = mappings_config.get('lifestyle_mapping', {})

# 所有映射的统一访问
ALL_MAPPINGS = {
    "collecting_base_info": BASE_INFO_MAPPING,
    "collecting_symptoms": SYMPTOM_MAPPING,
    "life_style": LIFESTYLE_MAPPING,
    "combined_info": COMBINED_INFO_MAPPING
}


# 辅助函数
def get_mapping_for_state(state_value: str):
    """根据状态值获取相应的字段映射"""
    return ALL_MAPPINGS.get(state_value, {})


def get_field_info(state_value: str, field_name: str):
    """获取指定状态和字段的详细信息"""
    mapping = get_mapping_for_state(state_value)
    return mapping.get(field_name, {})


def format_field_descriptions(state_value: str):
    """格式化字段描述，用于LLM提示"""
    mapping = get_mapping_for_state(state_value)
    descriptions = []

    for field, info in mapping.items():
        examples = ", ".join(info["examples"])
        descriptions.append(
            f"{field}({info['zh_name']}): {info['description']}，例如：{examples}"
        )

    return "\n- ".join([""] + descriptions)


def get_fields_by_importance(state_value: str, importance_level: str):
    """获取指定重要性级别的字段列表"""
    mapping = get_mapping_for_state(state_value)
    return [
        field for field, info in mapping.items()
        if info.get("importance") == importance_level
    ]
# src/dialogue/field_mappings.py
"""
医疗对话中使用的字段映射定义文件
提供详细的字段说明、中文映射和示例
"""
# 合并信息字段映射（包含基本信息和症状信息）
COMBINED_INFO_MAPPING = {
    # 症状信息（高优先级）
    "main": {
        "zh_name": "主要症状",
        "description": "患者当前最主要的不适症状",
        "examples": ["头痛", "腹痛", "发热", "咳嗽"],
        "importance": "high"
    },
    "duration": {
        "zh_name": "持续时间",
        "description": "症状已经持续了多长时间",
        "examples": ["2天", "一周左右", "3个小时", "反复发作半年"],
        "importance": "high"
    },
    "severity": {
        "zh_name": "严重程度",
        "description": "症状对日常生活和工作的影响程度",
        "examples": ["轻微，能忍受", "中等，影响工作", "严重，无法忍受"],
        "importance": "high"
    },

    # 基本信息（中优先级）
    "age": {
        "zh_name": "年龄",
        "description": "患者的实际年龄",
        "examples": ["23岁", "35", "四十多岁"],
        "importance": "high"
    },
    "gender": {
        "zh_name": "性别",
        "description": "患者的性别信息",
        "examples": ["男", "女", "男性", "女性"],
        "importance": "high"
    },

    # 症状详细信息（中低优先级）
    "pattern": {
        "zh_name": "症状模式",
        "description": "症状是持续性的还是间歇性的",
        "examples": ["持续疼痛", "每天早上加重", "间歇性发作", "运动后加重"],
        "importance": "medium"
    },
    "factors": {
        "zh_name": "加重/缓解因素",
        "description": "什么因素会加重或缓解症状",
        "examples": ["休息后好转", "进食后加重", "按压痛处有缓解", "天气变化时加重"],
        "importance": "medium"
    },
    "associated": {
        "zh_name": "伴随症状",
        "description": "主要症状以外的其他不适",
        "examples": ["同时有恶心", "伴有发热", "还有乏力", "无其他症状"],
        "importance": "medium"
    },

    # 基本病史信息（低优先级）
    "medical_history": {
        "zh_name": "病史",
        "description": "患者过去的疾病和手术记录",
        "examples": ["高血压", "糖尿病", "去年做过胆囊手术", "无"],
        "importance": "medium"
    },
    "allergy": {
        "zh_name": "过敏史",
        "description": "患者对药物或食物的过敏情况",
        "examples": ["对青霉素过敏", "海鲜过敏", "无过敏史"],
        "importance": "medium"
    },
    "medication": {
        "zh_name": "用药情况",
        "description": "患者目前正在服用的药物",
        "examples": ["服用降压药", "正在吃抗生素", "没有服用任何药物"],
        "importance": "medium"
    }
}

# 基本信息字段映射
BASE_INFO_MAPPING = {
    "age": {
        "zh_name": "年龄",
        "description": "患者的实际年龄",
        "examples": ["23岁", "35", "四十多岁"],
        "importance": "high"
    },
    "gender": {
        "zh_name": "性别",
        "description": "患者的性别信息",
        "examples": ["男", "女", "男性", "女性"],
        "importance": "high"
    },
    "medical_history": {
        "zh_name": "病史",
        "description": "患者过去的疾病和手术记录",
        "examples": ["高血压", "糖尿病", "去年做过胆囊手术", "无"],
        "importance": "medium"
    },
    "allergy": {
        "zh_name": "过敏史",
        "description": "患者对药物或食物的过敏情况",
        "examples": ["对青霉素过敏", "海鲜过敏", "无过敏史"],
        "importance": "medium"
    },
    "medication": {
        "zh_name": "用药情况",
        "description": "患者目前正在服用的药物",
        "examples": ["服用降压药", "正在吃抗生素", "没有服用任何药物"],
        "importance": "medium"
    }
}

# 症状信息字段映射
SYMPTOM_MAPPING = {
    "main_symptoms": {
        "zh_name": "主要症状",
        "description": "患者当前最主要的不适症状",
        "examples": ["头痛", "腹痛", "发热", "咳嗽"],
        "importance": "high"
    },
    "duration": {
        "zh_name": "持续时间",
        "description": "症状已经持续了多长时间",
        "examples": ["2天", "一周左右", "3个小时", "反复发作半年"],
        "importance": "high"
    },
    "severity": {
        "zh_name": "严重程度",
        "description": "症状对日常生活和工作的影响程度",
        "examples": ["轻微，能忍受", "中等，影响工作", "严重，无法忍受"],
        "importance": "high"
    },
    "pattern": {
        "zh_name": "症状模式",
        "description": "症状是持续性的还是间歇性的",
        "examples": ["持续疼痛", "每天早上加重", "间歇性发作", "运动后加重"],
        "importance": "medium"
    },
    "factors": {
        "zh_name": "加重/缓解因素",
        "description": "什么因素会加重或缓解症状",
        "examples": ["休息后好转", "进食后加重", "按压痛处有缓解", "天气变化时加重"],
        "importance": "medium"
    },
    "associated": {
        "zh_name": "伴随症状",
        "description": "主要症状以外的其他不适",
        "examples": ["同时有恶心", "伴有发热", "还有乏力", "无其他症状"],
        "importance": "medium"
    }
}

# 生活习惯字段映射
LIFESTYLE_MAPPING = {
    "sleep": {
        "zh_name": "睡眠情况",
        "description": "患者的睡眠时间和质量",
        "examples": ["每天7小时", "失眠", "睡眠质量差", "经常做梦"],
        "importance": "medium"
    },
    "diet": {
        "zh_name": "饮食习惯",
        "description": "患者的饮食规律和偏好",
        "examples": ["饮食不规律", "喜欢吃辛辣食物", "清淡饮食", "经常外食"],
        "importance": "medium"
    },
    "exercise": {
        "zh_name": "运动习惯",
        "description": "患者的运动频率和类型",
        "examples": ["每周跑步3次", "很少运动", "经常游泳", "工作太忙没时间运动"],
        "importance": "low"
    },
    "work": {
        "zh_name": "工作情况",
        "description": "患者的工作强度和压力",
        "examples": ["工作压力大", "长时间伏案工作", "轮班工作", "工作轻松"],
        "importance": "low"
    },
    "smoke_drink": {
        "zh_name": "烟酒习惯",
        "description": "患者的吸烟和饮酒情况",
        "examples": ["不吸烟不喝酒", "每天半包烟", "社交饮酒", "已戒烟5年"],
        "importance": "medium"
    }
}

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
# src/dialogue/utils.py
from typing import Dict, List, Tuple
from datetime import datetime


def format_medical_info(info: Dict) -> str:
    """格式化医疗信息"""
    sections = {
        "基本信息": ["age", "gender"],
        "病史信息": ["medical_history", "allergy", "medication"],
        "症状信息": ["main", "duration", "severity", "pattern", "factors", "associated"],
        "生活习惯": ["sleep", "diet", "exercise", "work", "smoke_drink"]
    }

    formatted = []
    for section, keys in sections.items():
        section_info = [f"{k}: {info.get(k, '未知')}" for k in keys if k in info]
        if section_info:
            formatted.append(f"{section}:\n" + "\n".join(section_info))

    return "\n\n".join(formatted)


def check_emergency(medical_info: Dict) -> Tuple[bool, str]:
    """紧急情况判断"""
    # 检查严重度
    severity = medical_info.get('severity')
    if severity and float(severity) >= 8:
        return True, "症状严重程度较高，建议及时就医"

    # 关键词检查
    emergency_conditions = {
        "严重疼痛": ["剧烈", "难忍", "剧痛"],
        "呼吸问题": ["呼吸困难", "胸闷", "窒息感"],
        "意识问题": ["意识不清", "昏迷", "晕厥"],
        "出血情况": ["大出血", "不止血"],
        "过敏反应": ["过敏", "喉咙肿胀"],
        "胸痛": ["胸痛", "心绞痛"]
    }

    for symptom_desc in medical_info.values():
        for condition, keywords in emergency_conditions.items():
            if any(k in str(symptom_desc) for k in keywords):
                return True, f"发现{condition}，建议立即就医"

    return False, ""


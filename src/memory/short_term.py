"""
短期记忆模块 - 管理当前对话和临时信息
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期记忆类，存储当前对话会话中的信息"""

    def __init__(self):
        """初始化短期记忆"""
        self.memory = {
            'current_dialogue': [],  # 当前对话历史
            'current_symptoms': [],  # 当前症状
            'temp_diagnosis': None,  # 临时诊断结果
            'entity_mentions': {},  # 实体提及(症状、药物等)
            'context_info': {},  # 上下文信息
        }
        self.last_update = datetime.now()

    def add_dialogue(self, role: str, content: str):
        """添加对话记录

        Args:
            role: 'doctor' 或 'patient'
            content: 对话内容
        """
        self.memory['current_dialogue'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.last_update = datetime.now()
        logger.debug(f"已添加对话: {role} - {content[:30]}...")

    def add_symptom(self, symptom: Dict[str, Any]):
        """添加症状记录

        Args:
            symptom: 症状信息字典，包含名称、严重程度、持续时间等
        """
        # 检查是否已存在相同症状，如果存在则更新
        symptom_name = symptom.get('name', '') if isinstance(symptom, dict) else symptom

        if isinstance(symptom, str):
            symptom = {'name': symptom}

        existing_symptoms = [s for s in self.memory['current_symptoms']
                             if (isinstance(s, dict) and s.get('name') == symptom_name) or s == symptom_name]

        if existing_symptoms:
            # 更新现有症状
            for existing in existing_symptoms:
                if isinstance(existing, dict) and isinstance(symptom, dict):
                    existing.update(symptom)
        else:
            # 添加新症状
            if isinstance(symptom, dict) and 'first_mentioned' not in symptom:
                symptom['first_mentioned'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.memory['current_symptoms'].append(symptom)

        self.last_update = datetime.now()
        logger.debug(f"已添加/更新症状: {symptom_name}")

    def set_temp_diagnosis(self, diagnosis: str):
        """设置临时诊断结果

        Args:
            diagnosis: 诊断结果文本
        """
        self.memory['temp_diagnosis'] = diagnosis
        self.last_update = datetime.now()
        logger.debug(f"已设置临时诊断: {diagnosis}")

    def add_entity_mention(self, entity_type: str, entity_name: str, context: str):
        """添加实体提及记录

        Args:
            entity_type: 实体类型(如'symptom', 'medication')
            entity_name: 实体名称
            context: 提及上下文
        """
        if entity_type not in self.memory['entity_mentions']:
            self.memory['entity_mentions'][entity_type] = {}

        if entity_name not in self.memory['entity_mentions'][entity_type]:
            self.memory['entity_mentions'][entity_type][entity_name] = []

        self.memory['entity_mentions'][entity_type][entity_name].append({
            'context': context,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.last_update = datetime.now()
        logger.debug(f"已添加实体提及: {entity_type} - {entity_name}")

    def update_context_info(self, key: str, value: Any):
        """更新上下文信息

        Args:
            key: 信息键名
            value: 信息值
        """
        self.memory['context_info'][key] = value
        self.last_update = datetime.now()
        logger.debug(f"已更新上下文信息: {key}")

    def get_current_dialogue(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.memory['current_dialogue']

    def get_current_symptoms(self) -> List[Dict]:
        """获取当前症状列表"""
        return self.memory['current_symptoms']

    def get_temp_diagnosis(self) -> Optional[str]:
        """获取临时诊断结果"""
        return self.memory['temp_diagnosis']

    def get_entity_mentions(self, entity_type: Optional[str] = None) -> Dict:
        """获取实体提及记录

        Args:
            entity_type: 可选的实体类型过滤

        Returns:
            所有或指定类型的实体提及记录
        """
        if entity_type:
            return self.memory['entity_mentions'].get(entity_type, {})
        return self.memory['entity_mentions']

    def get_context_info(self, key: Optional[str] = None) -> Any:
        """获取上下文信息

        Args:
            key: 可选的信息键名

        Returns:
            指定键的信息值或整个上下文信息字典
        """
        if key:
            return self.memory['context_info'].get(key)
        return self.memory['context_info']

    def clear(self):
        """清空短期记忆"""
        self.memory = {
            'current_dialogue': [],
            'current_symptoms': [],
            'temp_diagnosis': None,
            'entity_mentions': {},
            'context_info': {},
        }
        self.last_update = datetime.now()
        logger.info("已清空短期记忆")
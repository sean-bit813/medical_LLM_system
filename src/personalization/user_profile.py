# src/personalization/user_profile.py
"""
用户画像模块 - 存储和管理用户偏好数据
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# 配置日志
logger = logging.getLogger(__name__)


class UserProfile:
    """用户画像类，存储用户基本信息、病史和交互偏好"""

    def __init__(self, user_id: str):
        """初始化用户画像

        Args:
            user_id: 用户ID
        """
        self.user_id = user_id
        self.basic_info = {}  # 存储年龄、性别等基本信息
        self.medical_history = {}  # 既往病史/过敏史
        self.conversation_log = []  # 完整对话记录
        self.symptom_entities = {}  # 结构化症状信息
        self.preferences = {  # 交互偏好记录
            'communication_style': 'neutral',  # 沟通风格：professional/friendly/neutral
            'detail_level': 'normal'  # 信息详细程度：simple/normal/detailed
        }
        self.last_update = datetime.now()
        logger.info(f"已创建用户画像: {user_id}")

    def update_basic_info(self, info: Dict[str, Any]) -> None:
        """更新基本信息

        Args:
            info: 基本信息字典
        """
        self.basic_info.update(info)
        self.last_update = datetime.now()
        logger.debug(f"已更新用户基本信息: {self.user_id}")

    def update_medical_history(self, history: Dict[str, Any]) -> None:
        """更新医疗病史

        Args:
            history: 病史信息字典
        """
        self.medical_history.update(history)
        self.last_update = datetime.now()
        logger.debug(f"已更新用户病史: {self.user_id}")

    def add_conversation_entry(self, role: str, content: str) -> None:
        """添加对话记录

        Args:
            role: 'user' 或 'system'
            content: 对话内容
        """
        self.conversation_log.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.last_update = datetime.now()

    def add_symptom(self, symptom_name: str, details: Dict[str, Any] = None) -> None:
        """添加症状记录

        Args:
            symptom_name: 症状名称
            details: 症状详情
        """
        if details is None:
            details = {}

        if symptom_name not in self.symptom_entities:
            self.symptom_entities[symptom_name] = {
                'first_mentioned': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        self.symptom_entities[symptom_name].update(details)
        self.last_update = datetime.now()
        logger.debug(f"已添加/更新症状: {symptom_name}")

    def update_preference(self, preference_type: str, value: str) -> bool:
        """更新用户偏好设置

        Args:
            preference_type: 偏好类型
            value: 偏好值

        Returns:
            是否成功更新
        """
        if preference_type in self.preferences:
            self.preferences[preference_type] = value
            self.last_update = datetime.now()
            logger.debug(f"已更新用户偏好 {preference_type}: {value}")
            return True
        return False

    def get_communication_style(self) -> str:
        """获取用户沟通风格偏好"""
        return self.preferences.get('communication_style', 'neutral')

    def get_detail_level(self) -> str:
        """获取用户信息详细程度偏好"""
        return self.preferences.get('detail_level', 'normal')

    def get_symptom_history(self, symptom_name: Optional[str] = None) -> Dict:
        """获取症状历史

        Args:
            symptom_name: 可选的症状名称，用于获取特定症状

        Returns:
            症状历史字典
        """
        if symptom_name:
            return self.symptom_entities.get(symptom_name, {})
        return self.symptom_entities

    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """获取最近的对话记录

        Args:
            limit: 返回的最大记录数

        Returns:
            最近的对话记录列表
        """
        return self.conversation_log[-limit:] if self.conversation_log else []

    def to_dict(self) -> Dict[str, Any]:
        """将用户画像转换为字典，用于序列化

        Returns:
            用户画像的字典表示
        """
        return {
            'user_id': self.user_id,
            'basic_info': self.basic_info,
            'medical_history': self.medical_history,
            'symptom_entities': self.symptom_entities,
            'preferences': self.preferences,
            'last_update': self.last_update.strftime("%Y-%m-%d %H:%M:%S")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建用户画像对象

        Args:
            data: 用户画像字典数据

        Returns:
            UserProfile对象
        """
        profile = cls(data['user_id'])
        profile.basic_info = data.get('basic_info', {})
        profile.medical_history = data.get('medical_history', {})
        profile.symptom_entities = data.get('symptom_entities', {})
        profile.preferences = data.get('preferences', {
            'communication_style': 'neutral',
            'detail_level': 'normal'
        })

        return profile
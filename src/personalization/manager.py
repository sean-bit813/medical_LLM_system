# src/personalization/manager.py
"""
个性化管理器 - 管理用户画像和个性化交互
"""
import json
import logging
import os
from typing import Dict, Any, Optional, List

from .user_profile import UserProfile
from .preference_detector import PreferenceDetector
from .response_generator import ResponseGenerator

# 配置日志
logger = logging.getLogger(__name__)


class PersonalizationManager:
    """个性化管理器，管理用户画像和个性化交互"""

    def __init__(self, profiles_dir: str = "profiles"):
        """初始化个性化管理器

        Args:
            profiles_dir: 用户画像存储目录
        """
        self.profiles_dir = profiles_dir
        self.profiles = {}  # 用户ID到用户画像的映射
        self.preference_detector = PreferenceDetector()
        self.response_generator = ResponseGenerator()

        # 确保存储目录存在
        os.makedirs(profiles_dir, exist_ok=True)

        logger.info("个性化管理器初始化完成")

    def get_user_profile(self, user_id: str) -> UserProfile:
        """获取用户画像，如不存在则创建

        Args:
            user_id: 用户ID

        Returns:
            用户画像对象
        """
        # 检查内存缓存
        if user_id in self.profiles:
            return self.profiles[user_id]

        # 尝试从文件加载
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                profile = UserProfile.from_dict(profile_data)
                self.profiles[user_id] = profile
                logger.info(f"已从文件加载用户画像: {user_id}")
                return profile
            except Exception as e:
                logger.error(f"加载用户画像失败: {e}")

        # 创建新画像
        profile = UserProfile(user_id)
        self.profiles[user_id] = profile
        logger.info(f"已创建新用户画像: {user_id}")
        return profile

    def save_profile(self, user_id: str) -> bool:
        """保存用户画像到文件

        Args:
            user_id: 用户ID

        Returns:
            保存是否成功
        """
        if user_id not in self.profiles:
            logger.warning(f"保存用户画像失败: 用户ID {user_id} 不存在")
            return False

        profile = self.profiles[user_id]
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")

        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"已保存用户画像: {user_id}")
            return True
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")
            return False

    def update_profile_from_message(self, user_id: str, message: str,
                                    dialogue_history: List[Dict[str, Any]]) -> None:
        """从用户消息更新用户画像

        Args:
            user_id: 用户ID
            message: 用户消息
            dialogue_history: 对话历史
        """
        profile = self.get_user_profile(user_id)

        # 添加对话记录
        profile.add_conversation_entry('user', message)

        # 检测偏好信号
        preferences = self.preference_detector.detect_preferences(message, dialogue_history)

        # 更新偏好（只有在置信度较高时）
        if preferences.get('confidence', 0) > 0.7:
            communication_style = preferences.get('communication_style')
            if communication_style:
                profile.update_preference('communication_style', communication_style)

            detail_level = preferences.get('detail_level')
            if detail_level:
                profile.update_preference('detail_level', detail_level)

        # 保存更新后的画像
        self.save_profile(user_id)

    def process_input(self, user_id: str, user_input: str,
                      dialogue_history: List[Dict[str, Any]],
                      medical_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户输入，提取信息并更新用户画像

        Args:
            user_id: 用户ID
            user_input: 用户输入文本
            dialogue_history: 对话历史
            medical_info: 医疗信息

        Returns:
            处理结果，包含提取的信息和更新的偏好
        """
        profile = self.get_user_profile(user_id)

        # 更新对话记录
        profile.add_conversation_entry('user', user_input)

        # 检测偏好信号
        preferences = self.preference_detector.detect_preferences(user_input, dialogue_history)

        # 更新偏好（只有在置信度较高时）
        preference_updated = False
        if preferences.get('confidence', 0) > 0.7:
            communication_style = preferences.get('communication_style')
            if communication_style:
                profile.update_preference('communication_style', communication_style)
                preference_updated = True

            detail_level = preferences.get('detail_level')
            if detail_level:
                profile.update_preference('detail_level', detail_level)
                preference_updated = True

        # 更新医疗信息
        if 'symptoms' in medical_info:
            for symptom in medical_info['symptoms']:
                if isinstance(symptom, dict) and 'name' in symptom:
                    profile.add_symptom(symptom['name'], symptom)
                elif isinstance(symptom, str):
                    profile.add_symptom(symptom)

        # 更新基本信息
        basic_info_fields = ['age', 'gender', 'name']
        basic_info_updated = False
        for field in basic_info_fields:
            if field in medical_info and medical_info[field]:
                profile.basic_info[field] = medical_info[field]
                basic_info_updated = True

        # 更新病史
        medical_history_fields = ['medical_history', 'allergy', 'medication']
        medical_history_updated = False
        for field in medical_history_fields:
            if field in medical_info and medical_info[field]:
                profile.medical_history[field] = medical_info[field]
                medical_history_updated = True

        # 保存更新后的画像
        self.save_profile(user_id)

        # 返回处理结果
        return {
            'preference_updated': preference_updated,
            'basic_info_updated': basic_info_updated,
            'medical_history_updated': medical_history_updated,
            'detected_preferences': preferences
        }

    def generate_personalized_response(self, user_id: str, base_response: str,
                                       medical_info: Dict[str, Any]) -> str:
        """生成个性化响应

        Args:
            user_id: 用户ID
            base_response: 基础响应
            medical_info: 医疗信息

        Returns:
            个性化后的响应
        """
        profile = self.get_user_profile(user_id)

        # 添加对话记录
        profile.add_conversation_entry('system', base_response)

        # 添加个性化部分
        personalized_response = self.response_generator.add_personalized_parts(
            base_response, profile, medical_info
        )

        # 调整响应风格
        final_response = self.response_generator.adapt_response_style(
            personalized_response, profile
        )

        # 保存更新后的画像
        self.save_profile(user_id)

        return final_response
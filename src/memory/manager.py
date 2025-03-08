"""
记忆管理器 - 协调短期、中期和长期记忆系统
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .long_term import LongTermMemory

# 配置日志
logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器，协调三级记忆系统"""

    def __init__(self):
        """初始化记忆管理器"""
        self.short_term = ShortTermMemory()
        self.mid_term = MidTermMemory()
        self.long_term = LongTermMemory()
        self.current_patient_id = None
        logger.info("记忆管理器初始化完成")

    def start_new_consultation(self, patient_id: str):
        """开始新的问诊

        Args:
            patient_id: 患者ID
        """
        self.current_patient_id = patient_id

        # 重置短期记忆
        self.short_term = ShortTermMemory()

        # 尝试从中期和长期记忆中加载患者信息
        try:
            # 从中期记忆获取基本信息
            patient_info = self.mid_term.get_patient_info(patient_id)
            if patient_info:
                # 更新上下文信息
                for key, value in patient_info.items():
                    self.short_term.update_context_info(key, value)
                logger.info(f"已从中期记忆加载患者信息: {patient_id}")

            # 尝试从长期记忆获取历史信息
            long_term_info = self.long_term.get_patient_history(patient_id)
            if long_term_info and long_term_info.get('profile'):
                # 将关键历史信息添加到上下文
                profile = long_term_info['profile']
                self.short_term.update_context_info('long_term_profile', profile)
                logger.info(f"已从长期记忆加载患者档案: {patient_id}")

            # 记录过去的症状和诊断，以便后续对话引用
            self._load_past_symptoms_and_diagnoses(patient_id)

        except Exception as e:
            logger.error(f"加载患者历史信息失败: {e}")

        logger.info(f"已开始新的问诊: {patient_id}")

    def _load_past_symptoms_and_diagnoses(self, patient_id: str):
        """从中期记忆加载过去的症状和诊断

        Args:
            patient_id: 患者ID
        """
        try:
            # 获取最近的就诊记录
            consultations = self.mid_term.get_consultations(patient_id, limit=3)

            if consultations:
                past_symptoms = set()
                past_diagnoses = set()

                for consultation in consultations:
                    # 收集症状
                    symptoms = consultation.get('symptoms', [])
                    for symptom in symptoms:
                        if isinstance(symptom, dict) and 'name' in symptom:
                            past_symptoms.add(symptom['name'])
                        elif isinstance(symptom, str):
                            past_symptoms.add(symptom)

                    # 收集诊断
                    diagnosis = consultation.get('diagnosis')
                    if diagnosis:
                        past_diagnoses.add(diagnosis)

                # 更新到短期记忆中
                self.short_term.update_context_info('past_symptoms', list(past_symptoms))
                self.short_term.update_context_info('past_diagnoses', list(past_diagnoses))
                logger.info(f"已加载过去的症状和诊断: {len(past_symptoms)}个症状, {len(past_diagnoses)}个诊断")
        except Exception as e:
            logger.error(f"加载过去的症状和诊断失败: {e}")

    def add_dialogue(self, role: str, content: str):
        """添加对话内容

        Args:
            role: 'doctor' 或 'patient'
            content: 对话内容
        """
        self.short_term.add_dialogue(role, content)

    def add_symptom(self, symptom: Dict[str, Any]):
        """添加症状

        Args:
            symptom: 症状信息字典
        """
        self.short_term.add_symptom(symptom)

    def set_temp_diagnosis(self, diagnosis: str):
        """设置临时诊断

        Args:
            diagnosis: 诊断结果
        """
        self.short_term.set_temp_diagnosis(diagnosis)

    def save_consultation(self):
        """保存本次问诊记录到中期和长期记忆"""
        if not self.current_patient_id:
            logger.warning("保存问诊记录失败：未指定当前患者ID")
            return

        try:
            # 准备问诊数据
            consultation_data = {
                'dialogue': self.short_term.get_current_dialogue(),
                'symptoms': self.short_term.get_current_symptoms(),
                'diagnosis': self.short_term.get_temp_diagnosis(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 保存到中期记忆
            self.mid_term.add_consultation_record(self.current_patient_id, consultation_data)
            logger.info(f"已保存问诊记录到中期记忆: {self.current_patient_id}")

            # 检查是否有重要信息值得保存到长期记忆
            has_diagnosis = consultation_data['diagnosis'] is not None
            has_symptoms = len(consultation_data['symptoms']) > 0

            if has_diagnosis or has_symptoms:
                # 转换为长期记忆格式并保存
                long_term_data = {
                    'consultation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symptoms_summary': self._extract_symptoms_summary(consultation_data['symptoms']),
                    'diagnosis': consultation_data['diagnosis'] or "未确定诊断",
                    'key_dialogue_points': self._extract_key_dialogue_points()
                }

                self.long_term.add_medical_history(self.current_patient_id, long_term_data)
                logger.info(f"已保存问诊记录到长期记忆: {self.current_patient_id}")
            else:
                logger.info(f"问诊记录没有重要诊断或症状，不保存到长期记忆")

        except Exception as e:
            logger.error(f"保存问诊记录失败: {e}")

    def _extract_symptoms_summary(self, symptoms):
        """提取症状摘要信息

        Args:
            symptoms: 症状列表

        Returns:
            简化的症状列表
        """
        summary = []

        for symptom in symptoms:
            if isinstance(symptom, dict) and 'name' in symptom:
                # 保留症状名称和关键属性
                simplified = {
                    'name': symptom['name']
                }
                # 添加重要属性
                for key in ['severity', 'duration', 'pattern', 'location']:
                    if key in symptom:
                        simplified[key] = symptom[key]
                summary.append(simplified)
            elif isinstance(symptom, str):
                summary.append({'name': symptom})

        return summary

    def _extract_key_dialogue_points(self) -> List[str]:
        """从对话中提取关键信息点

        Returns:
            关键对话点列表
        """
        dialogue = self.short_term.get_current_dialogue()

        # 提取患者对话中可能包含关键信息的句子
        key_points = []

        for item in dialogue:
            if item['role'] == 'patient':
                content = item['content']

                # 简单启发式：句子中包含特定关键词可能更重要
                important_keywords = ['严重', '疼', '痛', '不适', '过敏', '曾经', '历史',
                                      '以前', '家族', '遗传', '不能', '失眠', '药物']

                # 按句子拆分
                sentences = content.split('。')
                for sentence in sentences:
                    # 检查是否包含关键词
                    if any(keyword in sentence for keyword in important_keywords):
                        clean_sentence = sentence.strip()
                        if clean_sentence and len(clean_sentence) > 3:  # 避免太短的句子
                            key_points.append(clean_sentence)

        # 限制数量，避免过多
        return key_points[:5]

    def retrieve_relevant_memory(self, query: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """检索与查询相关的记忆信息

        Args:
            query: 查询文本
            patient_id: 可选的患者ID过滤

        Returns:
            包含短期、中期和长期记忆检索结果的字典
        """
        patient_id = patient_id or self.current_patient_id
        if not patient_id:
            logger.warning("检索记忆信息失败：未指定患者ID")
            return {
                'short_term': {},
                'mid_term': {},
                'long_term': []
            }

        try:
            # 获取中期记忆
            consultations = self.mid_term.get_consultations(patient_id, limit=3)
            prescriptions = self.mid_term.get_prescriptions(patient_id, limit=3)

            # 获取长期记忆
            long_term_results = self.long_term.retrieve_info(query, patient_id, k=5)

            # 整合结果
            results = {
                'short_term': {
                    'dialogue': self.short_term.get_current_dialogue(),
                    'symptoms': self.short_term.get_current_symptoms(),
                    'diagnosis': self.short_term.get_temp_diagnosis(),
                    'context': self.short_term.get_context_info()
                },
                'mid_term': {
                    'consultations': consultations,
                    'prescriptions': prescriptions
                },
                'long_term': long_term_results
            }

            return results
        except Exception as e:
            logger.error(f"检索记忆信息失败: {e}")
            return {
                'short_term': {},
                'mid_term': {},
                'long_term': []
            }

    def add_patient_basic_info(self, patient_id: str, info: Dict[str, Any]):
        """添加患者基本信息到中期和长期记忆

        Args:
            patient_id: 患者ID
            info: 患者基本信息
        """
        if not patient_id:
            logger.warning("添加患者信息失败：未指定患者ID")
            return

        try:
            # 保存到中期记忆
            self.mid_term.add_patient_info(patient_id, info)

            # 保存到长期记忆
            self.long_term.add_patient_profile(patient_id, info)

            # 更新当前患者ID
            if not self.current_patient_id:
                self.current_patient_id = patient_id

            # 更新短期记忆上下文
            for key, value in info.items():
                self.short_term.update_context_info(key, value)

            logger.info(f"已添加患者基本信息: {patient_id}")
        except Exception as e:
            logger.error(f"添加患者基本信息失败: {e}")

    # 在MemoryManager中添加临时ID记忆迁移功能
    def migrate_memory(self, temp_id: str, permanent_id: str):
        """将临时ID的记忆迁移到永久ID

        Args:
            temp_id: 临时患者ID
            permanent_id: 永久患者ID
        """
        try:
            # 从中期记忆获取临时ID的记录
            temp_consultations = self.mid_term.get_consultations(temp_id)

            # 迁移每条记录到永久ID
            for consultation in temp_consultations:
                # 修改患者ID
                consultation['patient_id'] = permanent_id
                # 保存到永久ID的记录中
                self.mid_term.add_consultation_record(permanent_id, consultation)

            # 从长期记忆获取并迁移记录
            temp_history = self.long_term.get_patient_history(temp_id)
            if temp_history and temp_history.get('medical_history'):
                for history_item in temp_history['medical_history']:
                    self.long_term.add_medical_history(permanent_id, history_item)

            logger.info(f"已将临时ID {temp_id} 的记忆迁移到 {permanent_id}")
            return True
        except Exception as e:
            logger.error(f"记忆迁移失败: {e}")
            return False

    # 在MemoryManager中添加定期保存功能
    def periodic_save(self, turn_count: int):
        """根据对话轮次定期保存记忆

        Args:
            turn_count: 当前对话轮次
        """
        # 每10轮对话保存一次
        if turn_count % 10 == 0 and self.current_patient_id:
            self.save_consultation()
            logger.info(f"第{turn_count}轮对话，已自动保存记忆")


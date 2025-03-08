"""
中期记忆模块 - 使用Redis存储就诊记录
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 导入app_config中的Redis配置
try:
    from ..app_config import REDIS_CONFIG
except ImportError:
    logger.warning("无法导入REDIS_CONFIG，将使用默认配置")
    REDIS_CONFIG = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": "",
        "prefix": "medical_mid_term:"
    }

# 尝试导入redis
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis模块未安装，将使用内存字典作为后备存储")
    REDIS_AVAILABLE = False


class MidTermMemory:
    """中期记忆类，使用Redis存储就诊记录、处方等信息"""

    def __init__(self):
        """初始化中期记忆，连接Redis"""
        self.redis = None
        self.memory = {}  # 内存字典作为后备

        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(
                    host=REDIS_CONFIG.get('host', 'localhost'),
                    port=REDIS_CONFIG.get('port', 6379),
                    db=REDIS_CONFIG.get('db', 0),
                    password=REDIS_CONFIG.get('password', ''),
                    decode_responses=True  # 自动解码响应
                )
                # 测试连接
                self.redis.ping()
                self.prefix = REDIS_CONFIG.get('prefix', 'medical_mid_term:')
                self.ttl = REDIS_CONFIG.get('ttl', 2592000)  # 默认30天过期
                logger.info("成功连接到Redis服务器")
            except Exception as e:
                logger.error(f"Redis连接失败: {e}")
                self.redis = None
                logger.warning("将使用内存字典作为后备存储")

    def _get_patient_key(self, patient_id: str) -> str:
        """获取患者的Redis键

        Args:
            patient_id: 患者ID

        Returns:
            Redis键名
        """
        return f"{self.prefix}patient:{patient_id}"

    def _get_consultation_key(self, patient_id: str, consultation_id: str) -> str:
        """获取就诊记录的Redis键

        Args:
            patient_id: 患者ID
            consultation_id: 就诊ID

        Returns:
            Redis键名
        """
        return f"{self.prefix}consultation:{patient_id}:{consultation_id}"

    def add_patient_info(self, patient_id: str, patient_info: Dict[str, Any]):
        """添加或更新患者基本信息

        Args:
            patient_id: 患者ID
            patient_info: 患者基本信息字典
        """
        key = self._get_patient_key(patient_id)
        data = json.dumps(patient_info, ensure_ascii=False)

        try:
            if self.redis:
                # 使用Redis存储
                self.redis.set(key, data)
                self.redis.expire(key, self.ttl)
                logger.debug(f"已存储患者信息到Redis: {patient_id}")
            else:
                # 使用内存字典后备
                self.memory[key] = data
                logger.debug(f"已在内存中存储患者信息: {patient_id}")
        except Exception as e:
            logger.error(f"存储患者信息失败: {e}")

    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """获取患者基本信息

        Args:
            patient_id: 患者ID

        Returns:
            患者信息字典，如果不存在则返回None
        """
        key = self._get_patient_key(patient_id)

        try:
            if self.redis:
                # 从Redis获取
                data = self.redis.get(key)
            else:
                # 从内存字典获取
                data = self.memory.get(key)

            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"获取患者信息失败: {e}")
            return None

    def add_consultation_record(self, patient_id: str, consultation_data: Dict[str, Any]):
        """添加就诊记录

        Args:
            patient_id: 患者ID
            consultation_data: 就诊数据字典
        """
        # 生成唯一的就诊ID
        consultation_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{patient_id}"
        key = self._get_consultation_key(patient_id, consultation_id)

        # 添加时间戳
        if 'timestamp' not in consultation_data:
            consultation_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = json.dumps(consultation_data, ensure_ascii=False)

        try:
            if self.redis:
                # 使用Redis存储
                self.redis.set(key, data)
                self.redis.expire(key, self.ttl)

                # 更新患者的就诊记录索引
                index_key = f"{self.prefix}consultation_index:{patient_id}"
                self.redis.sadd(index_key, consultation_id)
                self.redis.expire(index_key, self.ttl)

                logger.debug(f"已存储就诊记录到Redis: {patient_id}, {consultation_id}")
            else:
                # 使用内存字典后备
                self.memory[key] = data

                # 更新患者的就诊记录索引
                index_key = f"{self.prefix}consultation_index:{patient_id}"
                if index_key not in self.memory:
                    self.memory[index_key] = set()
                if isinstance(self.memory[index_key], set):
                    self.memory[index_key].add(consultation_id)
                else:
                    self.memory[index_key] = {consultation_id}

                logger.debug(f"已在内存中存储就诊记录: {patient_id}, {consultation_id}")
        except Exception as e:
            logger.error(f"存储就诊记录失败: {e}")

    def get_consultations(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取患者的就诊记录

        Args:
            patient_id: 患者ID
            limit: 返回的最大记录数

        Returns:
            就诊记录列表，按时间降序排序
        """
        index_key = f"{self.prefix}consultation_index:{patient_id}"
        consultations = []

        try:
            # 获取就诊ID列表
            consultation_ids = set()
            if self.redis:
                # 从Redis获取就诊ID列表
                consultation_ids = self.redis.smembers(index_key)
            else:
                # 从内存字典获取就诊ID列表
                if index_key in self.memory:
                    if isinstance(self.memory[index_key], set):
                        consultation_ids = self.memory[index_key]
                    else:
                        consultation_ids = {self.memory[index_key]}

            # 获取每个就诊记录的详细信息
            for consultation_id in consultation_ids:
                key = self._get_consultation_key(patient_id, consultation_id)

                data = None
                if self.redis:
                    data = self.redis.get(key)
                else:
                    data = self.memory.get(key)

                if data:
                    try:
                        consultation = json.loads(data)
                        consultation['id'] = consultation_id
                        consultations.append(consultation)
                    except json.JSONDecodeError:
                        logger.error(f"JSON解析错误: {data}")

            # 按时间戳降序排序
            consultations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # 限制返回数量
            return consultations[:limit]
        except Exception as e:
            logger.error(f"获取就诊记录失败: {e}")
            return []

    def add_prescription(self, patient_id: str, prescription: Dict[str, Any]):
        """添加处方记录

        Args:
            patient_id: 患者ID
            prescription: 处方数据字典
        """
        # 生成唯一的处方ID
        prescription_id = f"rx_{datetime.now().strftime('%Y%m%d%H%M%S')}_{patient_id}"
        key = f"{self.prefix}prescription:{patient_id}:{prescription_id}"

        # 添加时间戳
        if 'timestamp' not in prescription:
            prescription['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = json.dumps(prescription, ensure_ascii=False)

        try:
            if self.redis:
                # 使用Redis存储
                self.redis.set(key, data)
                self.redis.expire(key, self.ttl)

                # 更新患者的处方索引
                index_key = f"{self.prefix}prescription_index:{patient_id}"
                self.redis.sadd(index_key, prescription_id)
                self.redis.expire(index_key, self.ttl)

                logger.debug(f"已存储处方到Redis: {patient_id}, {prescription_id}")
            else:
                # 使用内存字典后备
                self.memory[key] = data

                # 更新患者的处方索引
                index_key = f"{self.prefix}prescription_index:{patient_id}"
                if index_key not in self.memory:
                    self.memory[index_key] = set()
                if isinstance(self.memory[index_key], set):
                    self.memory[index_key].add(prescription_id)
                else:
                    self.memory[index_key] = {prescription_id}

                logger.debug(f"已在内存中存储处方: {patient_id}, {prescription_id}")
        except Exception as e:
            logger.error(f"存储处方失败: {e}")

    def get_prescriptions(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取患者的处方记录

        Args:
            patient_id: 患者ID
            limit: 返回的最大记录数

        Returns:
            处方记录列表，按时间降序排序
        """
        index_key = f"{self.prefix}prescription_index:{patient_id}"
        prescriptions = []

        try:
            # 获取处方ID列表
            prescription_ids = set()
            if self.redis:
                # 从Redis获取处方ID列表
                prescription_ids = self.redis.smembers(index_key)
            else:
                # 从内存字典获取处方ID列表
                if index_key in self.memory:
                    if isinstance(self.memory[index_key], set):
                        prescription_ids = self.memory[index_key]
                    else:
                        prescription_ids = {self.memory[index_key]}

            # 获取每个处方的详细信息
            for prescription_id in prescription_ids:
                key = f"{self.prefix}prescription:{patient_id}:{prescription_id}"

                data = None
                if self.redis:
                    data = self.redis.get(key)
                else:
                    data = self.memory.get(key)

                if data:
                    try:
                        prescription = json.loads(data)
                        prescription['id'] = prescription_id
                        prescriptions.append(prescription)
                    except json.JSONDecodeError:
                        logger.error(f"JSON解析错误: {data}")

            # 按时间戳降序排序
            prescriptions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # 限制返回数量
            return prescriptions[:limit]
        except Exception as e:
            logger.error(f"获取处方记录失败: {e}")
            return []
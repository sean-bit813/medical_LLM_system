"""
长期记忆模块 - 使用RAGFlow存储和检索患者长期信息
"""
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入RAGFlow配置
try:
    from ..app_config import RAGFLOW_CONFIG
except ImportError:
    logger.warning("无法导入RAGFLOW_CONFIG，将使用默认配置")
    RAGFLOW_CONFIG = {
        "api_url": "https://ragflow-api/api/v1",
        "api_key": "",
        "dataset_ids": [],
        "similarity_threshold": 0.2,
        "rerank_id": None
    }

# 尝试导入记忆系统配置
try:
    from ..app_config import MEMORY_CONFIG
except ImportError:
    logger.warning("无法导入MEMORY_CONFIG，将使用默认配置")
    MEMORY_CONFIG = {
        "short_term_max_dialogues": 20,
        "mid_term_expiry_days": 30,
        "long_term_min_relevance": 0.75
    }

# 尝试导入RAGFlow SDK
try:
    from ragflow_sdk import RAGFlow as RAGFlowSDK

    RAGFLOW_SDK_AVAILABLE = True
except ImportError:
    logger.warning("RAGFlow SDK未安装，将使用HTTP请求API代替")
    RAGFLOW_SDK_AVAILABLE = False


class LongTermMemory:
    """长期记忆类，使用RAGFlow存储和检索患者档案和病史"""

    def __init__(self, vector_dim: int = None):
        """初始化长期记忆

        Args:
            vector_dim: 向量维度，默认使用配置文件中的值
        """
        self.api_url = RAGFLOW_CONFIG.get('api_url', '')
        self.api_key = RAGFLOW_CONFIG.get('api_key', '')
        self.dataset_ids = RAGFLOW_CONFIG.get('dataset_ids', [])
        self.collection = "medical_long_term_memory"
        self.vector_dim = vector_dim or 512
        self.min_relevance = MEMORY_CONFIG.get('long_term_min_relevance', 0.75)

        # 本地缓存
        self.patient_profiles = {}  # 患者档案
        self.medical_history = {}  # 病史记录

        # 尝试初始化RAGFlow客户端
        self.rag_client = None
        self.dataset = None
        if RAGFLOW_SDK_AVAILABLE and self.api_key:
            try:
                self._init_ragflow_client()
            except Exception as e:
                logger.error(f"初始化RAGFlow客户端失败: {e}")

        logger.info("长期记忆初始化完成")

    def _init_ragflow_client(self):
        """初始化RAGFlow客户端"""
        if not RAGFLOW_SDK_AVAILABLE:
            logger.warning("RAGFlow SDK未安装，无法初始化客户端")
            return

        self.rag_client = RAGFlowSDK(
            api_key=self.api_key,
            base_url=self.api_url
        )

        # 确保数据集存在
        self._ensure_dataset_exists()

    def _ensure_dataset_exists(self):
        """确保长期记忆数据集存在"""
        if not self.rag_client:
            return

        try:
            # 检查是否已有医疗长期记忆数据集
            datasets = self.rag_client.list_datasets(name=self.collection)

            if datasets:
                self.dataset = datasets[0]
                # 保存数据集ID
                dataset_id = getattr(self.dataset, 'id', None)
                if dataset_id and dataset_id not in self.dataset_ids:
                    self.dataset_ids.append(dataset_id)
                logger.info(f"使用现有长期记忆数据集: {dataset_id}")
            else:
                # 创建新数据集
                self.dataset = self.rag_client.create_dataset(
                    name=self.collection,
                    description="医疗助手长期记忆存储",
                    language="Chinese"
                )
                # 保存数据集ID
                dataset_id = getattr(self.dataset, 'id', None)
                if dataset_id and dataset_id not in self.dataset_ids:
                    self.dataset_ids.append(dataset_id)
                logger.info(f"创建新的长期记忆数据集: {dataset_id}")
        except Exception as e:
            logger.error(f"确保数据集存在失败: {e}")

    def _make_api_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict:
        """发送API请求到RAGFlow

        Args:
            endpoint: API端点路径
            method: HTTP方法
            data: 请求数据

        Returns:
            API响应
        """
        url = f"{self.api_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=data)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, json=data)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"RAGFlow API请求失败: {e}")
            return {"error": str(e)}

    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本的向量嵌入

        在实际实现中，这应该调用RAGFlow的嵌入API。
        这里我们使用模拟的随机向量作为示例。

        Args:
            text: 文本内容

        Returns:
            嵌入向量
        """
        try:
            # 尝试调用RAGFlow API生成嵌入
            response = self._make_api_request(
                "embed",
                method="POST",
                data={"text": text}
            )

            if "error" not in response and "embedding" in response:
                return response["embedding"]

            # 如果API调用失败，使用随机向量替代
            logger.warning("使用随机向量替代真实嵌入")
            return np.random.rand(self.vector_dim).tolist()
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            # 返回随机向量作为后备
            return np.random.rand(self.vector_dim).tolist()

    def add_patient_profile(self, patient_id: str, profile_data: Dict[str, Any]):
        """添加患者档案

        Args:
            patient_id: 患者ID
            profile_data: 患者档案数据
        """
        # 更新本地缓存
        self.patient_profiles[patient_id] = profile_data

        # 如果没有RAGFlow客户端，则只保存到本地缓存
        if not self.rag_client and not self.api_key:
            logger.warning(f"RAGFlow未配置，患者档案只保存到本地缓存: {patient_id}")
            return

        # 转换为JSON字符串
        profile_text = json.dumps(profile_data, ensure_ascii=False)

        try:
            if RAGFLOW_SDK_AVAILABLE and self.dataset:
                # 使用SDK添加文档
                doc_name = f"profile_{patient_id}.json"
                blob = profile_text.encode('utf-8')

                # 上传文档
                self.dataset.upload_documents([{
                    'display_name': doc_name,
                    'blob': blob
                }])

                logger.info(f"已存储患者档案到RAGFlow: {patient_id}")
            else:
                # 使用API请求添加文档
                embedding = self._generate_embedding(profile_text)

                document = {
                    "id": f"profile_{patient_id}",
                    "text": profile_text,
                    "metadata": {
                        "patient_id": patient_id,
                        "type": "profile",
                        "timestamp": datetime.now().isoformat()
                    },
                    "embedding": embedding
                }

                # 存储到RAGFlow
                if self.dataset_ids:
                    dataset_id = self.dataset_ids[0]
                    self._make_api_request(
                        f"collections/{dataset_id}/documents",
                        method="POST",
                        data=document
                    )
                    logger.info(f"已通过API存储患者档案: {patient_id}")
        except Exception as e:
            logger.error(f"存储患者档案异常: {e}")

    def add_medical_history(self, patient_id: str, history_data: Dict[str, Any]):
        """添加病史记录

        Args:
            patient_id: 患者ID
            history_data: 病史数据
        """
        # 更新本地缓存
        if patient_id not in self.medical_history:
            self.medical_history[patient_id] = []
        self.medical_history[patient_id].append(history_data)

        # 如果没有RAGFlow客户端，则只保存到本地缓存
        if not self.rag_client and not self.api_key:
            logger.warning(f"RAGFlow未配置，病史记录只保存到本地缓存: {patient_id}")
            return

        # 转换为JSON字符串
        history_text = json.dumps(history_data, ensure_ascii=False)

        try:
            if RAGFLOW_SDK_AVAILABLE and self.dataset:
                # 使用SDK添加文档
                doc_name = f"history_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                blob = history_text.encode('utf-8')

                # 上传文档
                self.dataset.upload_documents([{
                    'display_name': doc_name,
                    'blob': blob
                }])

                logger.info(f"已存储病史记录到RAGFlow: {patient_id}")
            else:
                # 使用API请求添加文档
                history_id = f"history_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                embedding = self._generate_embedding(history_text)

                document = {
                    "id": history_id,
                    "text": history_text,
                    "metadata": {
                        "patient_id": patient_id,
                        "type": "medical_history",
                        "timestamp": datetime.now().isoformat()
                    },
                    "embedding": embedding
                }

                # 存储到RAGFlow
                if self.dataset_ids:
                    dataset_id = self.dataset_ids[0]
                    self._make_api_request(
                        f"collections/{dataset_id}/documents",
                        method="POST",
                        data=document
                    )
                    logger.info(f"已通过API存储病史记录: {history_id}")
        except Exception as e:
            logger.error(f"存储病史记录异常: {e}")

    def retrieve_info(self, query_text: str, patient_id: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """根据查询文本检索相关的记忆信息

        Args:
            query_text: 查询文本
            patient_id: 可选的患者ID过滤
            k: 返回的最大结果数

        Returns:
            相关记忆列表
        """
        # 如果未配置RAGFlow，只搜索本地缓存
        if not self.rag_client and not self.api_key:
            logger.warning("RAGFlow未配置，只搜索本地缓存")
            return self._search_local_cache(query_text, patient_id, k)

        try:
            results = []

            if RAGFLOW_SDK_AVAILABLE and self.dataset_ids:
                # 使用SDK检索
                query_params = {
                    "question": query_text,
                    "dataset_ids": self.dataset_ids,
                    "similarity_threshold": self.min_relevance,
                    "top_k": k,
                }

                if RAGFLOW_CONFIG.get('rerank_id'):
                    query_params["rerank_id"] = RAGFLOW_CONFIG.get('rerank_id')

                # 使用retrieve方法检索
                chunks = self.rag_client.retrieve(**query_params)

                # 处理结果
                for chunk in chunks:
                    try:
                        chunk_content = chunk.content if hasattr(chunk, 'content') else ""
                        chunk_meta = {}

                        # 尝试解析JSON内容
                        try:
                            content_obj = json.loads(chunk_content)
                            results.append({
                                "content": content_obj,
                                "metadata": chunk_meta,
                                "relevance": getattr(chunk, 'similarity', 0)
                            })
                        except json.JSONDecodeError:
                            # 如果不是有效的JSON，直接保存文本
                            results.append({
                                "content": chunk_content,
                                "metadata": chunk_meta,
                                "relevance": getattr(chunk, 'similarity', 0)
                            })
                    except Exception as inner_e:
                        logger.error(f"处理检索结果出错: {inner_e}")
            else:
                # 使用API检索
                # 准备查询参数
                query_vector = self._generate_embedding(query_text)

                query_params = {
                    "vector": query_vector,
                    "k": k,
                    "min_relevance": self.min_relevance
                }

                # 如果指定了患者ID，添加过滤条件
                if patient_id:
                    query_params["filter"] = {"metadata.patient_id": patient_id}

                # 发送搜索请求
                if self.dataset_ids:
                    dataset_id = self.dataset_ids[0]
                    response = self._make_api_request(
                        f"collections/{dataset_id}/search",
                        method="POST",
                        data=query_params
                    )

                    # 解析和格式化结果
                    if "error" not in response:
                        for item in response.get("results", []):
                            try:
                                text = item.get("text", "{}")
                                metadata = item.get("metadata", {})
                                relevance = item.get("relevance", 0)

                                # 尝试解析JSON文本
                                try:
                                    content = json.loads(text)
                                    results.append({
                                        "content": content,
                                        "metadata": metadata,
                                        "relevance": relevance
                                    })
                                except json.JSONDecodeError:
                                    # 如果不是有效的JSON，直接保存文本
                                    results.append({
                                        "content": text,
                                        "metadata": metadata,
                                        "relevance": relevance
                                    })
                            except Exception as item_e:
                                logger.error(f"处理检索结果项出错: {item_e}")

            # 如果RAGFlow检索没有结果，尝试本地缓存
            if not results:
                local_results = self._search_local_cache(query_text, patient_id, k)
                results.extend(local_results)

            return results
        except Exception as e:
            logger.error(f"检索记忆信息异常: {e}")
            # 尝试从本地缓存搜索
            return self._search_local_cache(query_text, patient_id, k)

    def _search_local_cache(self, query_text: str, patient_id: Optional[str] = None, k: int = 5) -> List[
        Dict[str, Any]]:
        """从本地缓存搜索相关信息

        Args:
            query_text: 查询文本
            patient_id: 可选的患者ID过滤
            k: 返回的最大结果数

        Returns:
            相关记忆列表
        """
        results = []

        # 如果指定了患者ID，只搜索该患者的信息
        if patient_id:
            # 添加患者档案
            if patient_id in self.patient_profiles:
                results.append({
                    "content": self.patient_profiles[patient_id],
                    "metadata": {"patient_id": patient_id, "type": "profile"},
                    "relevance": 1.0  # 完全匹配
                })

            # 添加病史记录
            if patient_id in self.medical_history:
                for history in self.medical_history[patient_id]:
                    results.append({
                        "content": history,
                        "metadata": {"patient_id": patient_id, "type": "medical_history"},
                        "relevance": 0.9  # 较高相关性
                    })
        else:
            # 搜索所有患者信息
            for pid, profile in self.patient_profiles.items():
                # 简单关键词匹配
                profile_text = json.dumps(profile, ensure_ascii=False)
                if any(term in profile_text for term in query_text.split()):
                    results.append({
                        "content": profile,
                        "metadata": {"patient_id": pid, "type": "profile"},
                        "relevance": 0.8  # 关键词匹配
                    })

            # 搜索病史记录
            for pid, histories in self.medical_history.items():
                for history in histories:
                    history_text = json.dumps(history, ensure_ascii=False)
                    if any(term in history_text for term in query_text.split()):
                        results.append({
                            "content": history,
                            "metadata": {"patient_id": pid, "type": "medical_history"},
                            "relevance": 0.7  # 关键词匹配
                        })

        # 按相关性排序
        results.sort(key=lambda x: x["relevance"], reverse=True)

        # 限制返回数量
        return results[:k]

    def get_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """获取患者的完整历史信息

        Args:
            patient_id: 患者ID

        Returns:
            包含档案和病史的综合信息
        """
        # 检索患者档案
        profile_query = f"患者{patient_id}档案"
        profile_results = self.retrieve_info(profile_query, patient_id, k=1)
        profile = profile_results[0].get("content", {}) if profile_results else {}

        # 检索患者病史
        history_query = f"患者{patient_id}病史"
        history_results = self.retrieve_info(history_query, patient_id, k=10)
        history = [item.get("content", {}) for item in history_results
                   if item.get("metadata", {}).get("type") == "medical_history"]

        # 整合结果
        result = {
            "patient_id": patient_id,
            "profile": profile,
            "medical_history": history
        }

        return result
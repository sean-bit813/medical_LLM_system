import os
import logging
from typing import Union, Dict, Any, List

from .kb import KnowledgeBase
from .ragflow_kb import RAGFlowKnowledgeBase
from ..app_config import RAGFLOW_CONFIG

logger = logging.getLogger(__name__)


class KnowledgeBaseFactory:
    """知识库工厂类，用于创建不同类型的知识库实例"""

    @staticmethod
    def create_knowledge_base(kb_type: str = "local", **kwargs) -> Union[KnowledgeBase, RAGFlowKnowledgeBase]:
        """
        创建并返回知识库实例

        Args:
            kb_type: 知识库类型，"local" 或 "ragflow"
            **kwargs: 传递给知识库构造函数的参数

        Returns:
            知识库实例
        """
        if kb_type.lower() == "ragflow":
            logger.info("创建RAGFlow知识库")

            # 从配置文件获取默认值
            api_url = kwargs.get("api_url", RAGFLOW_CONFIG.get("api_url"))
            api_key = kwargs.get("api_key", RAGFLOW_CONFIG.get("api_key"))
            dataset_ids = kwargs.get("dataset_ids", RAGFLOW_CONFIG.get("dataset_ids", []))

            return RAGFlowKnowledgeBase(
                api_url=api_url,
                api_key=api_key,
                dataset_ids=dataset_ids
            )
        else:
            logger.info("创建本地知识库")

            # 本地知识库参数
            index_path = kwargs.get("index_path")

            kb = KnowledgeBase()

            # 如果提供了索引路径，尝试加载
            if index_path and os.path.exists(index_path):
                logger.info(f"加载本地知识库索引: {index_path}")
                kb.load_index(index_path)

            # 如果提供了数据路径，加载数据
            csv_path = kwargs.get("csv_path")
            if csv_path and os.path.exists(csv_path):
                logger.info(f"加载知识库数据: {csv_path}")
                kb.load_data(csv_path)

                # 如果提供了索引路径，保存索引
                if index_path and not os.path.exists(index_path):
                    logger.info(f"保存知识库索引: {index_path}")
                    kb.save_index(index_path)

            return kb
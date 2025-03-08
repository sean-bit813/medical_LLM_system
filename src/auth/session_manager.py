"""
会话管理器 - 处理用户会话和令牌
"""
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器，处理用户会话和令牌验证"""

    def __init__(self, session_timeout: int = 3600):
        """初始化会话管理器

        Args:
            session_timeout: 会话超时时间（秒）
        """
        self.sessions = {}  # 会话字典，键为会话ID
        self.session_timeout = session_timeout  # 会话超时时间（秒）
        logger.info("会话管理器初始化完成")

    def create_session(self, username: str) -> str:
        """创建新会话

        Args:
            username: 用户名

        Returns:
            会话ID
        """
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())

        # 创建会话记录
        self.sessions[session_id] = {
            "username": username,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "data": {}
        }

        logger.info(f"为用户 {username} 创建会话: {session_id}")
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """验证会话是否有效

        Args:
            session_id: 会话ID

        Returns:
            会话是否有效
        """
        # 检查会话是否存在
        if session_id not in self.sessions:
            return False

        # 检查会话是否过期
        session = self.sessions[session_id]
        elapsed = (datetime.now() - session["last_activity"]).total_seconds()

        if elapsed > self.session_timeout:
            # 会话已过期，删除
            del self.sessions[session_id]
            logger.info(f"会话已过期: {session_id}")
            return False

        # 更新最后活动时间
        session["last_activity"] = datetime.now()
        return True

    def get_username(self, session_id: str) -> Optional[str]:
        """获取会话关联的用户名

        Args:
            session_id: 会话ID

        Returns:
            用户名，如会话无效则返回None
        """
        if self.validate_session(session_id):
            return self.sessions[session_id]["username"]
        return None

    def get_session_data(self, session_id: str, key: str) -> Any:
        """获取会话数据

        Args:
            session_id: 会话ID
            key: 数据键名

        Returns:
            会话数据值，如不存在则返回None
        """
        if self.validate_session(session_id):
            return self.sessions[session_id]["data"].get(key)
        return None

    def set_session_data(self, session_id: str, key: str, value: Any) -> bool:
        """设置会话数据

        Args:
            session_id: 会话ID
            key: 数据键名
            value: 数据值

        Returns:
            设置是否成功
        """
        if self.validate_session(session_id):
            self.sessions[session_id]["data"][key] = value
            return True
        return False

    def end_session(self, session_id: str) -> bool:
        """结束会话

        Args:
            session_id: 会话ID

        Returns:
            操作是否成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"会话已结束: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """清理所有过期会话

        Returns:
            清理的会话数量
        """
        expired_count = 0
        current_time = datetime.now()

        # 找出所有过期会话
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if (current_time - session["last_activity"]).total_seconds() > self.session_timeout
        ]

        # 删除过期会话
        for sid in expired_sessions:
            del self.sessions[sid]
            expired_count += 1

        if expired_count > 0:
            logger.info(f"已清理{expired_count}个过期会话")

        return expired_count
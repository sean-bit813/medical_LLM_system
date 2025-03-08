"""
用户管理器 - 处理用户注册、登录和密码验证
"""
import json
import hashlib
import logging
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)


class UserManager:
    """用户管理器，处理用户注册、认证和个人信息管理"""

    def __init__(self, users_file: str = "users.json"):
        """初始化用户管理器

        Args:
            users_file: 用户数据文件路径
        """
        self.users_file = users_file
        self.users = self._load_users()
        logger.info(f"用户管理器初始化完成，已加载{len(self.users)}个用户")

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """从文件加载用户数据

        Returns:
            用户数据字典，键为用户ID
        """
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"加载用户数据失败: {e}")
                return {}
        return {}

    def _save_users(self) -> bool:
        """保存用户数据到文件

        Returns:
            保存是否成功
        """
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
            return True
        except IOError as e:
            logger.error(f"保存用户数据失败: {e}")
            return False

    def _hash_password(self, password: str) -> str:
        """使用SHA-256哈希密码

        Args:
            password: 原始密码

        Returns:
            密码哈希值
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str, user_info: Dict[str, Any] = None) -> Tuple[bool, str]:
        """注册新用户

        Args:
            username: 用户名
            password: 密码
            user_info: 其他用户信息

        Returns:
            (成功标志, 消息)
        """
        # 检查用户名是否已存在
        if username in self.users:
            return False, "用户名已存在"

        # 创建用户记录
        user_data = {
            "password_hash": self._hash_password(password),
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "info": user_info or {}
        }

        # 添加用户
        self.users[username] = user_data

        # 保存到文件
        if self._save_users():
            logger.info(f"用户注册成功: {username}")
            return True, "注册成功"
        else:
            # 保存失败，回滚
            del self.users[username]
            return False, "注册失败：无法保存用户数据"

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """验证用户凭据

        Args:
            username: 用户名
            password: 密码

        Returns:
            (认证成功标志, 消息)
        """
        # 检查用户是否存在
        if username not in self.users:
            return False, "用户名不存在"

        # 验证密码
        password_hash = self._hash_password(password)
        if password_hash != self.users[username]["password_hash"]:
            return False, "密码不正确"

        # 更新登录时间
        self.users[username]["last_login"] = datetime.now().isoformat()
        self._save_users()

        logger.info(f"用户登录成功: {username}")
        return True, "登录成功"

    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息

        Args:
            username: 用户名

        Returns:
            用户信息字典，如不存在则返回None
        """
        if username in self.users:
            return self.users[username].get("info", {})
        return None

    def update_user_info(self, username: str, user_info: Dict[str, Any]) -> bool:
        """更新用户信息

        Args:
            username: 用户名
            user_info: 新的用户信息

        Returns:
            更新是否成功
        """
        if username in self.users:
            self.users[username]["info"] = user_info
            return self._save_users()
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """修改用户密码

        Args:
            username: 用户名
            old_password: 旧密码
            new_password: 新密码

        Returns:
            (修改成功标志, 消息)
        """
        # 先验证旧密码
        auth_result, msg = self.authenticate(username, old_password)
        if not auth_result:
            return False, msg

        # 更新密码
        self.users[username]["password_hash"] = self._hash_password(new_password)

        # 保存更改
        if self._save_users():
            logger.info(f"用户密码已更改: {username}")
            return True, "密码已更改"
        else:
            return False, "密码更改失败：无法保存用户数据"
import json
import os


class ConfigLoader:
    """配置加载器，用于从JSON文件加载配置"""

    @staticmethod
    def load_json_config(filename):
        """
        加载JSON配置文件

        Args:
            filename: 配置文件名，相对于config目录

        Returns:
            加载的JSON对象
        """
        # 确定配置文件路径
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        file_path = os.path.join(config_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {filename} 不存在，请检查路径: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {filename} 格式错误，请检查JSON语法")
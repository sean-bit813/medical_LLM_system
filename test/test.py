import unittest
from io import StringIO
from unittest.mock import patch, MagicMock
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.qa_system import pipeline, init_knowledge_base


class TestMedicalQASystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化知识库
        init_knowledge_base(
            csv_path=r"D:\LLM\medical_LLM_system\data\knowledge_base\sample_IM_5000-6000_utf8.csv",
            index_path=r"D:\LLM\medical_LLM_system\data\vector_store\sample_IM_5000-6000_utf8.index"
        )

    # 捕捉print内容
    def setUp(self):
        self.held_output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.held_output

    def tearDown(self):
        sys.stdout = self.old_stdout
        print(self.held_output.getvalue())

    def test_complete_medical_query(self):
        """测试完整的医疗查询"""
        messages = [
            {"role": "user", "content": "我想知道如何预防心脏病？我没有家族病史，但是有高血压"}
        ]
        response = pipeline(messages)
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

    def test_incomplete_query_followup(self):
        """测试不完整查询的反问流程"""
        messages = [
            {"role": "user", "content": "我最近咳嗽得厉害"}
        ]
        response = pipeline(messages)
        self.assertIn("多久", response)  # 检查是否包含时间相关的反问

    def test_multi_turn_conversation(self):
        """测试多轮对话"""
        messages = [
            {"role": "user", "content": "我经常失眠"},
            {"role": "assistant", "content": "请问您失眠的具体表现是什么？入睡困难还是容易醒？"},
            {"role": "user", "content": "主要是睡不着，每天要躺床上2小时才能入睡"}
        ]
        response = pipeline(messages)
        self.assertIsNotNone(response)

    def test_casual_chat(self):
        """测试闲聊模式"""
        messages = [
            {"role": "user", "content": "今天天气真好"}
        ]
        response = pipeline(messages)
        self.assertIsNotNone(response)

    @patch('src.qa_system.kb.search')
    def test_knowledge_base_error(self, mock_search):
        """测试知识库检索异常"""
        mock_search.side_effect = Exception("Knowledge base error")
        messages = [
            {"role": "user", "content": "治疗肥胖症胃绕道手术步骤，无其他基础疾病以及过敏，年龄29， 性别男， BMI45，术前正常"}
        ]
        response = pipeline(messages)
        self.assertIsNotNone(response)

    """
    def test_empty_query(self):
       
        messages = [
            {"role": "user", "content": ""}
        ]
        with self.assertRaises(Exception):
            pipeline(messages)
    """


if __name__ == '__main__':
    unittest.main(verbosity=2)

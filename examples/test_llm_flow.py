# examples/test_llm_flow.py
import os
import sys
import json
import logging
from datetime import datetime
import copy

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dialogue.manager import DialogueManager
from src.knowledge.kb import KnowledgeBase
from src.dialogue.states import DialogueState
from src.llm.api import generate_simple_response
from src.prompts.medical_prompts import LLM_FLOW_PROMPTS
from src.dialogue.field_mappings import format_field_descriptions

# 为LLM API添加中间件，捕获所有API调用
original_generate_simple_response = generate_simple_response


def generate_simple_response_with_logging(prompt, system_prompt=None, temperature=0.1, max_tokens=200):
    """带日志记录的API调用包装器"""
    logger.info(f"LLM API调用 | 提示词: {prompt[:100]}...")
    response = original_generate_simple_response(prompt, system_prompt, temperature, max_tokens)
    logger.info(f"LLM API响应 | {response[:100]}...")
    return response


# 替换原函数以启用日志记录
import src.llm.api

src.llm.api.generate_simple_response = generate_simple_response_with_logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("llm_flow_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_llm_flow")


class LLMFlowTester:
    """LLM对话流程测试工具"""

    def __init__(self, debug_mode=True, verbose=True):
        """初始化测试工具"""
        self.debug_mode = debug_mode
        self.verbose = verbose  # 是否打印详细信息
        self.manager = self._init_system()
        self.conversation_log = []
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _init_system(self):
        """初始化系统"""
        # 初始化知识库（可以使用测试专用的小型知识库）
        kb = KnowledgeBase()
        try:
            kb.load_index('../data/vector_store/sample_IM_5000-6000_utf8.index')
            logger.info("成功加载知识库索引")
        except Exception as e:
            logger.warning(f"加载知识库索引失败: {e}")
            # 如果需要可以在这里加载数据

        # 初始化对话管理器
        manager = DialogueManager(kb)
        manager.set_use_llm_flow(True)  # 确保使用LLM流程

        return manager

    def _log_step(self, step_name, data):
        """记录测试步骤"""
        if self.debug_mode:
            logger.info(f"[{step_name}] {json.dumps(data, ensure_ascii=False, default=str)}")

        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "data": data
        })

        if self.verbose and step_name != "LLM API调用":
            print(f"\n[{step_name}]")
            self._pretty_print(data)

    def _pretty_print(self, data, indent=2):
        """美化打印数据"""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) or isinstance(v, list):
                    print(" " * indent + f"{k}:")
                    self._pretty_print(v, indent + 2)
                else:
                    print(" " * indent + f"{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) or isinstance(item, list):
                    self._pretty_print(item, indent + 2)
                else:
                    print(" " * indent + f"- {item}")
        else:
            print(" " * indent + str(data))

    def process_message(self, user_input):
        """处理用户消息并记录各步骤信息"""
        # 记录用户输入
        self._log_step("用户输入", {"message": user_input})

        # 保存处理前的状态
        before_state = self.manager.context.state.value
        before_info = copy.deepcopy(self.manager.context.medical_info)
        self._log_step("处理前状态", {
            "state": before_state,
            "turn_count": self.manager.context.turn_count,
            "collected_info": before_info
        })

        # 获取当前Flow的详细信息
        if self.manager.current_flow:
            flow_info = {
                "flow_type": self.manager.current_flow.__class__.__name__,
                "required_info": self.manager.current_flow.required_info,
                "current_index": self.manager.current_flow.current_index,
                "use_llm_flow": self.manager.current_flow.use_llm_flow
            }
            self._log_step("当前Flow信息", flow_info)

        # 监听LLM调用
        self._setup_llm_monitoring()

        # 处理消息
        response = self.manager.process_message(user_input)

        # 保存处理后的状态
        after_state = self.manager.context.state.value
        after_info = copy.deepcopy(self.manager.context.medical_info)

        # 记录处理结果
        self._log_step("处理结果", {
            "response": response,
            "state_changed": before_state != after_state,
            "before_state": before_state,
            "after_state": after_state
        })

        # 详细记录字段变化
        field_changes = {}
        for k in set(list(after_info.keys()) + list(before_info.keys())):
            if k not in before_info:
                field_changes[k] = {
                    "change_type": "added",
                    "new_value": after_info[k]
                }
            elif k not in after_info:
                field_changes[k] = {
                    "change_type": "removed",
                    "old_value": before_info[k]
                }
            elif before_info[k] != after_info[k]:
                field_changes[k] = {
                    "change_type": "modified",
                    "old_value": before_info[k],
                    "new_value": after_info[k]
                }

        if field_changes:
            self._log_step("字段变化", field_changes)

        # 更新后的Flow信息
        if self.manager.current_flow:
            flow_info = {
                "flow_type": self.manager.current_flow.__class__.__name__,
                "required_info": self.manager.current_flow.required_info,
                "current_index": self.manager.current_flow.current_index,
                "use_llm_flow": self.manager.current_flow.use_llm_flow
            }
            self._log_step("更新后Flow信息", flow_info)

        # 检查是否提取了症状严重程度
        if "severity" in after_info and (
                "severity" not in before_info or before_info["severity"] != after_info["severity"]):
            self._log_step("严重程度评估", {
                "severity": after_info["severity"],
                "severity_numeric": self._try_parse_severity(after_info["severity"])
            })

        # 如果是紧急情况，记录详情
        if after_state == DialogueState.REFERRAL.value and before_state != DialogueState.REFERRAL.value:
            self._log_step("紧急情况检测", {
                "is_emergency": True,
                "advice": after_info.get("emergency_advice", "需要紧急处理")
            })

        return response

    def _setup_llm_monitoring(self):
        """设置LLM调用监控"""
        # 可以在此添加对LLM调用的更多监控
        pass

    def _try_parse_severity(self, severity_str):
        """尝试将严重程度转换为数字"""
        try:
            return int(severity_str)
        except (ValueError, TypeError):
            return None

    def run_interactive_test(self):
        """运行交互式测试"""
        print("=" * 50)
        print("LLM对话流程测试工具 (测试ID: {})".format(self.test_id))
        print("=" * 50)
        print("输入'exit'或'quit'退出测试")
        print("输入'debug on/off'开启/关闭调试模式")
        print("输入'verbose on/off'开启/关闭详细输出")
        print("输入'save'保存当前对话日志")
        print("输入'status'查看当前状态")
        print("输入'context'查看完整上下文")
        print("-" * 50)

        print("医疗助手: 您好，我是您的医疗助手。有什么可以帮您?")

        while True:
            user_input = input("\n患者: ").strip()

            # 处理特殊命令
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
            elif user_input.lower() == 'debug on':
                self.debug_mode = True
                print("调试模式已开启")
                continue
            elif user_input.lower() == 'debug off':
                self.debug_mode = False
                print("调试模式已关闭")
                continue
            elif user_input.lower() == 'verbose on':
                self.verbose = True
                print("详细输出已开启")
                continue
            elif user_input.lower() == 'verbose off':
                self.verbose = False
                print("详细输出已关闭")
                continue
            elif user_input.lower() == 'save':
                self._save_conversation_log()
                print(f"对话日志已保存到 conversation_{self.test_id}.json")
                continue
            elif user_input.lower() == 'status':
                self._print_current_status()
                continue
            elif user_input.lower() == 'context':
                self._print_full_context()
                continue
            elif user_input.lower().startswith('test '):
                self._handle_test_command(user_input[5:])
                continue

            # 处理正常消息
            response = self.process_message(user_input)
            print(f"\n医疗助手: {response}")

            # 检查对话是否结束
            if self.manager.context.state == DialogueState.ENDED:
                print("\n对话已结束")
                self._save_conversation_log()
                break

    def _handle_test_command(self, command):
        """处理测试命令"""
        parts = command.split()
        if not parts:
            print("无效的测试命令")
            return

        if parts[0] == 'extract':
            # 测试字段提取
            text = " ".join(parts[1:]) if len(parts) > 1 else input("请输入要提取字段的文本: ")
            state_value = self.manager.context.state.value
            result = test_field_extraction(text, state_value)
            print(f"\n提取结果: {result}")
        elif parts[0] == 'severity':
            # 测试严重程度评估
            text = " ".join(parts[1:]) if len(parts) > 1 else input("请输入要评估严重程度的文本: ")
            result = test_severity_assessment(text)
            print(f"\n严重程度评估结果: {result}")
        elif parts[0] == 'emergency':
            # 测试紧急情况判断
            text = " ".join(parts[1:]) if len(parts) > 1 else input("请输入要评估紧急情况的文本: ")
            result = test_emergency_assessment(text)
            print(f"\n紧急情况评估结果: {result}")
        else:
            print(f"未知的测试命令: {parts[0]}")

    def _print_current_status(self):
        """打印当前状态信息"""
        print("\n--- 当前状态信息 ---")
        print(f"当前状态: {self.manager.context.state.value}")
        print(f"对话轮次: {self.manager.context.turn_count}")

        if self.manager.current_flow:
            print(f"当前Flow: {self.manager.current_flow.__class__.__name__}")
            print(f"需要收集的信息: {self.manager.current_flow.required_info}")
            print(f"当前索引: {self.manager.current_flow.current_index}")
            print(f"使用LLM流程: {self.manager.current_flow.use_llm_flow}")

        print("已收集信息:")
        for k, v in self.manager.context.medical_info.items():
            print(f"  - {k}: {v}")
        print("-------------------\n")

    def _print_full_context(self):
        """打印完整上下文信息"""
        print("\n=== 完整上下文信息 ===")
        context_data = {
            "state": self.manager.context.state.value,
            "turn_count": self.manager.context.turn_count,
            "start_time": self.manager.context.start_time,
            "last_update": self.manager.context.last_update,
            "user_info": self.manager.context.user_info,
            "medical_info": self.manager.context.medical_info
        }
        self._pretty_print(context_data)

        if self.manager.current_flow:
            print("\n当前Flow详细信息:")
            flow_data = {
                "class": self.manager.current_flow.__class__.__name__,
                "state": self.manager.current_flow.state.value,
                "required_info": self.manager.current_flow.required_info,
                "current_index": self.manager.current_flow.current_index,
                "use_llm_flow": self.manager.current_flow.use_llm_flow
            }
            self._pretty_print(flow_data)
        print("=====================\n")

    def _save_conversation_log(self):
        """保存对话日志"""
        filename = f"conversation_{self.test_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_log, f, ensure_ascii=False, indent=2, default=str)


# 辅助函数：测试从文本提取字段
def test_field_extraction(text, state="collecting_symptoms"):
    """测试从文本中提取字段"""
    # 获取字段描述
    field_descriptions = format_field_descriptions(state)

    # 准备提示词
    prompt_template = LLM_FLOW_PROMPTS[state]["info_extraction_template"]
    prompt = prompt_template.format(
        user_response=text,
        field_descriptions=field_descriptions
    )

    # 调用LLM
    logger.info(f"测试字段提取 | 文本: {text} | 状态: {state}")
    result = generate_simple_response(prompt)
    logger.info(f"字段提取结果 | {result}")

    return result


# 辅助函数：测试严重程度评估
def test_severity_assessment(text):
    """测试严重程度评估"""
    # 准备提示词
    prompt_template = LLM_FLOW_PROMPTS["severity_assessment_template"]
    prompt = prompt_template.format(user_response=text)

    # 调用LLM
    logger.info(f"测试严重程度评估 | 文本: {text}")
    result = generate_simple_response(prompt)
    logger.info(f"严重程度评估结果 | {result}")

    return result


# 辅助函数：测试紧急情况判断
def test_emergency_assessment(text, collected_info=""):
    """测试紧急情况判断"""
    # 准备提示词
    prompt_template = LLM_FLOW_PROMPTS["emergency_assessment_template"]
    prompt = prompt_template.format(
        user_response=text,
        collected_info=collected_info
    )

    # 调用LLM
    logger.info(f"测试紧急情况判断 | 文本: {text}")
    result = generate_simple_response(prompt)
    logger.info(f"紧急情况判断结果 | {result}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM对话流程测试工具")
    parser.add_argument('--mode', choices=['interactive', 'extract', 'severity', 'emergency'],
                        default='interactive', help='测试模式')
    parser.add_argument('--text', type=str, help='用于提取或评估的文本')
    parser.add_argument('--state', choices=['collecting_base_info', 'collecting_symptoms', 'life_style'],
                        default='collecting_symptoms', help='对话状态')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    args = parser.parse_args()

    if args.mode == 'interactive':
        tester = LLMFlowTester(verbose=args.verbose)
        tester.run_interactive_test()
    elif args.mode == 'extract':
        if not args.text:
            args.text = input("请输入要提取字段的文本: ")
        result = test_field_extraction(args.text, args.state)
        print(f"\n提取结果:\n{result}")
    elif args.mode == 'severity':
        if not args.text:
            args.text = input("请输入要评估严重程度的文本: ")
        result = test_severity_assessment(args.text)
        print(f"\n严重程度评估结果: {result}")
    elif args.mode == 'emergency':
        if not args.text:
            args.text = input("请输入要评估紧急情况的文本: ")
        collected = input("已收集的信息(可选): ")
        result = test_emergency_assessment(args.text, collected)
        print(f"\n紧急情况评估结果: {result}")
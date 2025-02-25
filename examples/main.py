# examples/main.py
import os
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dialogue.manager import DialogueManager
from src.knowledge.kb import KnowledgeBase
from src.config import DIALOGUE_CONFIG


# 加载或初始化向量存储
def init_knowledge_base(csv_path=None, index_path=None):
    """初始化或加载知识库"""
    global kb
    kb = KnowledgeBase()

    if index_path and os.path.exists(index_path):
        kb.load_index(index_path)
    elif csv_path:
        kb.load_data(csv_path)
        if index_path:
            kb.save_index(index_path)


def init_system(use_llm_flow=None):
    """
    初始化系统

    Args:
        use_llm_flow: 是否使用LLM驱动的流程，None表示使用配置文件中的设置
    """
    # 初始化知识库
    init_knowledge_base(
        csv_path='../data/knowledge_base/sample_IM_5000-6000_utf8.csv',
        index_path='../data/vector_store/sample_IM_5000-6000_utf8.index'
    )

    # 初始化对话管理器
    manager = DialogueManager(kb)

    # 设置是否使用LLM驱动的流程
    if use_llm_flow is not None:
        manager.set_use_llm_flow(use_llm_flow)
    else:
        manager.set_use_llm_flow(DIALOGUE_CONFIG.get("use_llm_flow", True))

    return manager


def main(use_llm_flow=None):
    """
    主程序入口

    Args:
        use_llm_flow: 是否使用LLM驱动的流程，None表示使用配置文件中的设置
    """
    manager = init_system(use_llm_flow)
    print("医疗助手： 您好,我是您的医疗助手。有什么可以帮您？")

    while True:
        user_input = input("患者: ").strip()
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break

        response = manager.process_message(user_input)
        print(f"医疗助手: {response}")

        # 打印调试信息
        if manager.context.state.value == 'ended':
            print("\n对话结束,感谢您的使用\n")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='医疗问答系统')
    parser.add_argument('--llm', action='store_true', help='使用LLM驱动的对话流程')
    parser.add_argument('--no-llm', dest='llm', action='store_false', help='不使用LLM驱动的对话流程')
    parser.set_defaults(llm=None)

    args = parser.parse_args()
    main(use_llm_flow=args.llm)
# examples/main.py
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dialogue.manager import DialogueManager
from src.knowledge.factory import KnowledgeBaseFactory
from src.app_config import DIALOGUE_CONFIG, RAGFLOW_CONFIG


# 初始化或加载知识库
def init_knowledge_base(use_ragflow=False, csv_path=None, index_path=None):
    """初始化或加载知识库"""
    if use_ragflow:
        # 使用RAGFlow作为知识库
        kb = KnowledgeBaseFactory.create_knowledge_base(
            kb_type="ragflow"
        )
        print("使用RAGFlow知识库")
        return kb
    else:
        # 使用本地知识库
        kb = KnowledgeBaseFactory.create_knowledge_base(
            kb_type="local",
            csv_path=csv_path,
            index_path=index_path
        )
        print(f"使用本地知识库")
        return kb


def init_system(use_llm_flow=None, use_ragflow=False):
    """
    初始化系统

    Args:
        use_llm_flow: 是否使用LLM驱动的流程，None表示使用配置文件中的设置
        use_ragflow: 是否使用RAGFlow知识库
    """
    # 初始化知识库
    if use_ragflow:
        kb = init_knowledge_base(use_ragflow=True)
    else:
        kb = init_knowledge_base(
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


def main(use_llm_flow=None, use_ragflow=False):
    """
    主程序入口

    Args:
        use_llm_flow: 是否使用LLM驱动的流程，None表示使用配置文件中的设置
        use_ragflow: 是否使用RAGFlow知识库
    """
    manager = init_system(use_llm_flow, use_ragflow)
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
    parser = argparse.ArgumentParser(description='医疗问答系统')
    parser.add_argument('--llm', action='store_true', help='使用LLM驱动的对话流程')
    parser.add_argument('--no-llm', dest='llm', action='store_false', help='不使用LLM驱动的对话流程')
    parser.add_argument('--ragflow', action='store_true', help='使用RAGFlow知识库')
    parser.set_defaults(llm=None, ragflow=False)

    args = parser.parse_args()
    main(use_llm_flow=args.llm, use_ragflow=args.ragflow)
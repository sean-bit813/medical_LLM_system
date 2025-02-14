# main.py
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dialogue.manager import DialogueManager
from src.knowledge.kb import KnowledgeBase


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


def init_system():
    # 初始化知识库
    init_knowledge_base(
        csv_path='../data/knowledge_base/sample_IM_5000-6000_utf8.csv',
        index_path='../data/vector_store/sample_IM_5000-6000_utf8.index'
    )

    # data_sources = [
    # {'type': 'csv', 'path': 'data/knowledge_base/medical_qa.csv'},
    # {'type': 'medical_book', 'path': 'data/knowledge_base/textbook.json'},
    # {'type': 'qa', 'path': 'data/knowledge_base/qa_data.json'}
    # ]
    # kb.load_data_from_multiple_sources(data_sources)

    # 初始化对话管理器
    return DialogueManager(kb)


def main():
    manager = init_system()
    print("医疗助手： 您好,我是您的医疗助手。我将逐步指引您完成问诊流程，接下来会询问您一些基本信息，请回复“开始”（或除“退出”外任何词汇）进行咨询，回复“退出”将离开本次咨询")

    while True:
        user_input = input("患者: ").strip()
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break

        response = manager.process_message(user_input)
        print(f"医疗助手: {response}")

        # 打印调试信息
        if manager.context.state.value == 'ended':
            print("\n对话结束,感谢您的使用\n")
            #print(manager.get_context_summary())
            break


if __name__ == "__main__":
    main()

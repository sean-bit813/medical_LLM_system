import os
from sop_flows import sop_library
from prompts import prompts
from llm_api import call_doubao_api
from knowledge_base import KnowledgeBase


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

# 基于上下文的查询改写
def rewrite_query_with_context(messages):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])
    query = messages[-1]['content']
    prompt = prompts["rewrite_query_with_context"].format(context=context, query=query)
    messages_query_rewrite = [{"role": "user", "content": prompt}]
    return call_doubao_api(messages_query_rewrite, max_tokens=50)

# 意图识别
def identify_intent(query):
    prompt = prompts["identify_intent"].format(query=query)
    messages_intent = [{"role": "user", "content": prompt}]
    intent = call_doubao_api(messages_intent, max_tokens=10)
    print("原始意图：", intent)
    intent = intent.strip() if intent.strip() in prompts["check_and_complete_info"] else '闲聊'
    return intent

# 差异化信息补全判断并反问
def check_and_complete_info(query, intent):
    prompt = prompts["check_and_complete_info"][intent].format(query=query)
    messages_ask = [{"role": "user", "content": prompt}]
    completion = call_doubao_api(messages_ask, max_tokens=50)
    print("原始的completed_query: ", completion)
    if "None" not in completion:
        return completion.replace("反问：", "").replace("反问:", "").replace("反问", "").strip(), True
    return query, False

# 修改generate_response函数，增加知识检索功能
def generate_response(messages, sop_steps, is_chat=False): #is_chat判断是否是闲聊
    # 构建对话上下文
    conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # 如果不是闲聊模式，添加知识库检索
    knowledge_context = ""
    if not is_chat:
        try:
            current_query = messages[-1]['content']
            relevant_docs = kb.search(current_query, k=3)
            knowledge_context = "\n".join([doc['text'] for doc in relevant_docs])
            print("检索到的相关知识：", knowledge_context)
        except Exception as e:
            print(f"知识库检索错误: {e}")

    # 构建prompt
    context_text = f"对话历史：\n{conversation_context}"
    if knowledge_context:
        context_text = f"相关医学知识：\n{knowledge_context}\n\n{context_text}"

    final_prompt = prompts["generate_response"].format(
        steps=" -> ".join(sop_steps),
        context=context_text
    )

    messages_query_rewrite = [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user", "content": final_prompt},
    ]
    return call_doubao_api(messages_query_rewrite, max_tokens=100)


def pipeline(messages):

    rewritten_query = rewrite_query_with_context(messages)
    print("rewritten_query：", rewritten_query)
    intent = identify_intent(rewritten_query)
    print("intent: ", intent)
    sop_steps = sop_library[intent]
    if (intent == "闲聊"):
        response = generate_response(messages, sop_steps, is_chat=True)
        print("最终生成回答：", response)
        return response

    completed_query, is_follow_up = check_and_complete_info(rewritten_query, intent)
    print("completed_query: ", completed_query, is_follow_up)

    if is_follow_up:
        response = completed_query  # 生成反问，提供更多选项或填空
    else:
        response = generate_response(messages, sop_steps, is_chat=False)

    print("最终生成回答：", response)
    return response


# 使用示例
if (__name__ == "__main__"):
    # 初始化知识库
    init_knowledge_base(
        csv_path='../data/knowledge_base/sample_IM_5000-6000_utf8.csv',
        index_path='../data/vector_store/sample_IM_5000-6000_utf8.index'
    )

    # 测试问答
    # 1.不完整提问，需追加反问
    messages_1 = [
        {"role": "user", "content": "我最近总是感觉头痛，有什么建议吗"},
        {"role": "assistant", "content": "请问你头痛的情况持续了多久？"},
        {"role": "user", "content": "大概已经有一周了"}  # 正常case
    ]
    # 2. 完整提问，不需追加反问，生成医疗建议
    messages_2 = [
        {"role": "user", "content": "我想知道如何预防心脏病"},
        {"role": "assistant", "content": "请问你有没有家族遗传史或者其他健康问题？"},
        {"role": "user", "content": "家族史没有，我本人有高血压的问题"}  # 正常case
    ]
    print(pipeline(messages_2))
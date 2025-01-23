import os
from openai import OpenAI

# 配置豆包API
# 可参考官方文档：https://www.volcengine.com/docs/82379/1302008
doubao_api_key = open("D:\LLM\doubao_api_key.txt", "r").read()
doubao_base_url = "https://ark.cn-beijing.volces.com/api/v3"
# doubao-pro-32k
model_pod = open("D:\LLM\doubao_endpoint.txt", "r").read()

client = OpenAI(api_key=doubao_api_key, base_url=doubao_base_url)

def call_doubao_api(messages, max_tokens=500):
    # print("messages: ", messages)
    completion = client.chat.completions.create(
        model = model_pod,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content

if(__name__=="__main__"):
    messages = [
        {"role": "user", "content": "我最近总是感觉头痛，有什么建议吗"},
        {"role": "assistant", "content": "请问你头痛的情况持续了多久？"},
        {"role": "user", "content": "大概已经有一周了"}  # 正常case
    ]
    response = call_doubao_api(messages)
    print(response)
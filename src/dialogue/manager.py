# src/dialogue/manager.py
from typing import Dict, Optional, List
from datetime import datetime

from .utils import format_medical_info
from ..prompts.medical_prompts import MEDICAL_PROMPTS
from .states import DialogueState, StateContext
from .flows import FLOW_MAPPING
from ..config import DIALOGUE_CONFIG
from ..llm.api import generate_response
from ..knowledge.kb import KnowledgeBase


class DialogueManager:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.context = StateContext(
            state=DialogueState.INITIAL,
            user_info={},
            medical_info={},
            start_time=datetime.now()
        )
        self.current_flow = None
        self.kb = knowledge_base

    def _check_timeout(self) -> bool:
        elapsed = (datetime.now() - self.context.start_time).seconds
        return elapsed > DIALOGUE_CONFIG["timeout"]

    def _check_max_turns(self) -> bool:
        return self.context.turn_count >= DIALOGUE_CONFIG["max_turns"]

    def _should_end_conversation(self) -> bool:
        return (self._check_timeout() or
                self._check_max_turns() or
                self.context.state == DialogueState.ENDED)

    def _transition_state(self) -> None:
        if not self.current_flow:
            return

        next_state = self.current_flow.get_next_state(self.context)
        if next_state and next_state != self.context.state:
            self.context.state = next_state
            self.current_flow = (FLOW_MAPPING[next_state]() if next_state in FLOW_MAPPING else None)

    def _format_final_response(self) -> str:
        if self._check_timeout():
            return "对话时间已超时,建议重新开始咨询。"
        elif self._check_max_turns():
            return "已达到最大对话轮次,建议总结当前信息并考虑就医。"
        return "感谢您的咨询,祝您身体健康!"

    def _get_relevant_knowledge(self, query: str) -> str:
        """检索相关知识"""
        try:
            results = self.kb.search(query, k=3)
            return "\n".join([doc['text'] for doc in results])
        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return ""

    def _prepare_response_context(self, message: str) -> None:
        """准备生成响应所需的上下文"""
        if self.context.state in [DialogueState.DIAGNOSIS,
                                  DialogueState.MEDICAL_ADVICE,
                                  DialogueState.REFERRAL,
                                  DialogueState.EDUCATION]:
            query = f"{self.context.medical_info.get('main', '')} {message}"
            knowledge_content = self._get_relevant_knowledge(query)
            formatted_info = format_medical_info(self.context.medical_info)

            self.context.medical_info.update({
                'relevant_knowledge': knowledge_content,
                'formatted_info': formatted_info
            })

    def process_message(self, message: str) -> (str, None):
        if self._should_end_conversation():
            return self._format_final_response()

        self.context.update(turn_count=self.context.turn_count + 1)

        # 初始状态处理

        if self.context.state == DialogueState.INITIAL:
            self.context.state = DialogueState.COLLECTING_BASE_INFO
            self.current_flow = FLOW_MAPPING[DialogueState.COLLECTING_BASE_INFO]()
            # return MEDICAL_PROMPTS["initial"]

        # 处理消息
        if self.current_flow:

            is_emergency = self.current_flow.process_response(message, self.context)

            if is_emergency: # 更改当前flow
                self.current_flow = FLOW_MAPPING[DialogueState.REFERRAL]()

            # diagnosis, medical_advice, referral, education阶段只需要输出
            response = None
            if self.context.state in [DialogueState.DIAGNOSIS,
                                      DialogueState.MEDICAL_ADVICE,
                                      DialogueState.REFERRAL,
                                      DialogueState.EDUCATION]:
                self._prepare_response_context(message)
                response = generate_response(self.context)

            # 转换状态
            self._transition_state()

            # 检查是否还有问题
            next_question = None
            if self.current_flow:
                next_question = self.current_flow.get_next_question(self.context)

            #output = ""
            #if self.context.medical_info and len(self.context.medical_info) > 0:
             #   for index, (key, value) in enumerate(self.context.medical_info.items()):
              #      output += "KEY:" + str(key) + "VALUE:" + str(value)

            if response is None and next_question is None:
                return "已收集到您的信息，正在生成下一阶段建议，输入任意内容继续"
            return next_question if next_question else response + "\n 正在生成下一阶段建议，输入任意内容继续"

        return "抱歉，当前无法处理您的请求。"

# src/dialogue/manager.py
from typing import Dict, Optional, List
from datetime import datetime
import logging

from .utils import format_medical_info
from ..prompts.medical_prompts import MEDICAL_PROMPTS
from .states import DialogueState, StateContext
from .flows import FLOW_MAPPING
from ..app_config import DIALOGUE_CONFIG, RAGFLOW_CONFIG
from ..llm.api import generate_response
#from ..knowledge.kb import KnowledgeBase
from ..knowledge.ragflow_kb import RAGFlowKnowledgeBase
from ..config.loader import ConfigLoader

# 设置日志
logger = logging.getLogger(__name__)


class DialogueManager:
    def __init__(self, knowledge_base: RAGFlowKnowledgeBase):
        self.context = StateContext(
            state=DialogueState.INITIAL,
            user_info={},
            medical_info={},
            start_time=datetime.now()
        )
        self.current_flow = None
        self.kb = knowledge_base
        self.use_llm_flow = True  # 默认启用LLM驱动的对话流程

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
            logger.info(f"状态转换: {self.context.state.value} -> {next_state.value}")
            self.context.state = next_state
            self.current_flow = FLOW_MAPPING[next_state]() if next_state in FLOW_MAPPING else None
            # 为新创建的Flow设置LLM开关
            if self.current_flow and hasattr(self.current_flow, "use_llm_flow"):
                self.current_flow.use_llm_flow = self.use_llm_flow

    def _format_final_response(self) -> str:
        if self._check_timeout():
            return "对话时间已超时,建议重新开始咨询。"
        elif self._check_max_turns():
            return "已达到最大对话轮次,建议总结当前信息并考虑就医。"
        return "感谢您的咨询,祝您身体健康!"

    def _get_relevant_knowledge(self, query: str) -> str:
        """检索相关知识"""
        try:
            results = self.kb.search(query, k=5, similarity_threshold= RAGFLOW_CONFIG["similarity_threshold"], rerank_id=RAGFLOW_CONFIG["rerank_id"])

            # 处理返回结果的格式 - 兼容不同的知识库实现
            formatted_results = []
            for doc in results:
                # 统一处理不同知识库返回的结果格式
                if isinstance(doc, dict):
                    text = doc.get('text', '')
                    if text:
                        formatted_results.append(text)
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"知识库检索错误: {e}")
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

    def process_message(self, message: str) -> str:
        """处理用户消息，返回系统回复"""
        if self._should_end_conversation():
            self.context.state = DialogueState.ENDED
            return self._format_final_response()

        self.context.turn_count += 1
        logger.info(f"处理消息: {message}, 当前轮次: {self.context.turn_count}, 当前状态: {self.context.state.value}")

        # 初始状态处理
        if self.context.state == DialogueState.INITIAL:
            logger.info("从初始状态转换到基本信息收集状态")
            self.context.state = DialogueState.COLLECTING_COMBINED_INFO
            self.current_flow = FLOW_MAPPING[DialogueState.COLLECTING_COMBINED_INFO]()
            # 设置LLM流程开关
            if hasattr(self.current_flow, "use_llm_flow"):
                self.current_flow.use_llm_flow = self.use_llm_flow
            logger.info(f"初始化Flow: {self.current_flow.__class__.__name__}")

            # 对于第一次交互，直接返回欢迎问题
            if self.context.turn_count == 1:
                return "您好，我是您的医疗助手。请问您有什么不舒服的地方吗？"

        # 处理消息
        if self.current_flow:
            logger.info(f"当前Flow: {self.current_flow.__class__.__name__}, use_llm_flow={self.current_flow.use_llm_flow}")

            # 处理用户回复
            is_emergency = self.current_flow.process_response(message, self.context)

            if is_emergency:  # 更改当前flow
                logger.info("检测到紧急情况，转换到转诊流程")
                self.context.state = DialogueState.REFERRAL
                self.current_flow = FLOW_MAPPING[DialogueState.REFERRAL]()
                # 设置LLM流程开关
                if hasattr(self.current_flow, "use_llm_flow"):
                    self.current_flow.use_llm_flow = self.use_llm_flow

            # diagnosis, medical_advice, referral, education阶段只需要输出
            response = None
            if self.context.state in [DialogueState.DIAGNOSIS,
                                      DialogueState.MEDICAL_ADVICE,
                                      DialogueState.REFERRAL,
                                      DialogueState.EDUCATION]:
                self._prepare_response_context(message)
                response = generate_response(self.context)

            # 获取下一个问题
            next_question = None
            if not response and self.current_flow:
                # 不立即转换状态，而是先获取下一个问题
                next_question = self.current_flow.get_next_question(self.context)
                logger.info(f"下一个问题: {next_question}")

                # 如果有下一个问题，直接返回它
                if next_question:
                    return next_question

            # 如果没有下一个问题，才考虑转换状态
            if not next_question:
                self._transition_state()

                # 如果状态已转换，获取新状态的第一个问题
                if self.current_flow:
                    next_question = self.current_flow.get_next_question(self.context)
                    logger.info(f"状态转换后的下一个问题: {next_question}")

            if response is None and next_question is None:
                return "已收集到您的信息，正在生成下一阶段建议，输入任意内容继续"
            return next_question if next_question else response + "\n 正在生成下一阶段建议，输入任意内容继续"

        return "抱歉，当前无法处理您的请求。"

    def set_use_llm_flow(self, use_llm: bool) -> None:
        """设置是否使用LLM驱动的流程"""
        logger.info(f"设置use_llm_flow={use_llm}")
        self.use_llm_flow = use_llm
        # 更新当前Flow的设置
        if self.current_flow and hasattr(self.current_flow, "use_llm_flow"):
            self.current_flow.use_llm_flow = use_llm
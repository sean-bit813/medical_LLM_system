# src/dialogue/manager.py
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import logging

from .utils import format_medical_info
from ..memory import MemoryManager
from ..prompts.medical_prompts import MEDICAL_PROMPTS
from .states import DialogueState, StateContext
from .flows import FLOW_MAPPING
from ..app_config import DIALOGUE_CONFIG, RAGFLOW_CONFIG
from ..llm.api import generate_response
#from ..knowledge.kb import KnowledgeBase
from ..knowledge.ragflow_kb import RAGFlowKnowledgeBase
from ..config.loader import ConfigLoader
from ..auth.user_manager import UserManager
from ..auth.session_manager import SessionManager
from ..nlu.entity_recognition import symptom_entity_recognition
from ..nlu.intent_detection import detect_intent, is_emergency_intent
from ..nlu.context_analyzer import ContextAnalyzer

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
        self.memory_manager = MemoryManager()
        # 初始化用户和会话管理
        self.user_manager = UserManager()
        self.session_manager = SessionManager()
        # 添加NLU组件：初始化上下文分析器
        self.context_analyzer = ContextAnalyzer(self.memory_manager)

        logger.info("DialogueManager初始化完成，已创建记忆管理器")

    def register_user(self, username: str, password: str, user_info: Dict[str, Any] = None) -> Tuple[bool, str]:
        """注册新用户

        Args:
            username: 用户ID/用户名
            password: 密码
            user_info: 用户个人信息

        Returns:
            (注册成功标志, 消息)
        """
        return self.user_manager.register(username, password, user_info)

    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[str]]:
        """用户登录

        Args:
            username: 用户ID/用户名
            password: 密码

        Returns:
            (登录成功标志, 消息, 会话ID)
        """
        # 验证用户凭据
        auth_result, msg = self.user_manager.authenticate(username, password)

        if auth_result:
            # 创建新会话
            session_id = self.session_manager.create_session(username)

            # 设置当前患者ID
            self.context.user_info['patient_id'] = username

            # 初始化记忆系统
            self.memory_manager.start_new_consultation(username)

            # 记录用户信息
            user_info = self.user_manager.get_user_info(username)
            if user_info:
                self.memory_manager.add_patient_basic_info(username, user_info)

            return True, "登录成功", session_id

        return False, msg, None

    def validate_session(self, session_id: str) -> bool:
        """验证会话是否有效

        Args:
            session_id: 会话ID

        Returns:
            会话是否有效
        """
        is_valid = self.session_manager.validate_session(session_id)

        if is_valid:
            # 获取用户名并设置为患者ID
            username = self.session_manager.get_username(session_id)
            if username:
                self.context.user_info['patient_id'] = username

        return is_valid

    def logout_user(self, session_id: str) -> bool:
        """用户登出

        Args:
            session_id: 会话ID

        Returns:
            操作是否成功
        """
        # 保存当前会话到记忆系统
        patient_id = self.context.user_info.get('patient_id')
        if patient_id:
            self.memory_manager.save_consultation()

        # 结束会话
        return self.session_manager.end_session(session_id)

    def process_message_with_session(self, message: str, session_id: str) -> Tuple[bool, str]:
        """处理带会话ID的消息

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            (处理成功标志, 回复消息)
        """
        # 验证会话
        if not self.validate_session(session_id):
            return False, "会话已过期，请重新登录"

        # 处理消息
        response = self.process_message(message)

        return True, response

    def _get_or_create_patient_id(self) -> str:
        """获取患者ID

        优先从会话获取，其次从用户信息获取，最后创建临时ID

        Returns:
            患者ID字符串
        """
        # 优先从会话获取用户名作为患者ID
        if hasattr(self, 'session_manager') and hasattr(self, 'current_session_id'):
            username = self.session_manager.get_username(self.current_session_id)
            if username:
                # 确保上下文中也保存了患者ID
                self.context.user_info['patient_id'] = username
                logger.debug(f"从会话获取患者ID: {username}")
                return username

        # 其次从上下文用户信息中获取
        patient_id = self.context.user_info.get('patient_id')
        if patient_id:
            logger.debug(f"从上下文获取患者ID: {patient_id}")
            return patient_id

        # 最后创建临时ID
        temp_id = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.context.user_info['patient_id'] = temp_id
        logger.debug(f"创建临时患者ID: {temp_id}")
        return temp_id

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
            # 构建更智能的查询
            main_symptom = self.context.medical_info.get('main', '')
            patient_id = self._get_or_create_patient_id()

            # 构建增强查询，包含主要症状和当前消息
            query = f"{main_symptom} {message}"

            # 检索相关知识库信息
            knowledge_content = self._get_relevant_knowledge(query)

            # 检索相关记忆
            memory_results = self.memory_manager.retrieve_relevant_memory(query, patient_id)

            # 增强查询上下文
            self._enhance_context_with_memory(memory_results)

            # 格式化医疗信息
            formatted_info = format_medical_info(self.context.medical_info)

            # 提取过去的相关症状和诊断
            past_symptoms = []
            past_diagnoses = []

            if 'mid_term' in memory_results and 'consultations' in memory_results['mid_term']:
                for consult in memory_results['mid_term']['consultations']:
                    if 'symptoms' in consult:
                        for symptom in consult['symptoms']:
                            symptom_name = symptom.get('name', symptom) if isinstance(symptom, dict) else symptom
                            if symptom_name not in past_symptoms:
                                past_symptoms.append(symptom_name)

                    if 'diagnosis' in consult and consult['diagnosis']:
                        if consult['diagnosis'] not in past_diagnoses:
                            past_diagnoses.append(consult['diagnosis'])

            # 更新上下文医疗信息
            self.context.medical_info.update({
                'relevant_knowledge': knowledge_content,
                'formatted_info': formatted_info,
                'past_symptoms': past_symptoms,
                'past_diagnoses': past_diagnoses,
                'consultation_history': memory_results['mid_term'].get('consultations', [])
            })

    def _enhance_context_with_memory(self, memory_data):
        """使用记忆数据增强上下文"""
        # 提取短期记忆中的上下文信息
        if 'short_term' in memory_data and 'context' in memory_data['short_term']:
            context = memory_data['short_term']['context']

            # 添加过去症状到医疗信息
            if 'past_symptoms' in context:
                self.context.medical_info['past_symptoms'] = context['past_symptoms']

            # 添加过去诊断到医疗信息
            if 'past_diagnoses' in context:
                self.context.medical_info['past_diagnoses'] = context['past_diagnoses']

        # 提取中期记忆中的最近就诊记录
        if 'mid_term' in memory_data and 'consultations' in memory_data['mid_term']:
            consultations = memory_data['mid_term']['consultations']
            if consultations:
                # 取最近一次就诊记录中的症状作为参考
                self.context.medical_info['past_consultation'] = consultations[0]

        # 提取长期记忆中的见解
        if 'long_term' in memory_data and memory_data['long_term']:
            insights = []
            for insight in memory_data['long_term']:
                content = insight.get('content')
                if content:
                    insights.append(content)

            # 添加到医疗信息
            if insights:
                self.context.medical_info['long_term_insights'] = insights


    def process_message(self, message: str) -> str:
        """处理用户消息，返回系统回复"""
        # 1. 检查是否应该结束对话 (保持原有代码不变)
        if self._should_end_conversation():
            self.context.state = DialogueState.ENDED
            # 结束时保存对话到记忆系统
            patient_id = self._get_or_create_patient_id()
            self.memory_manager.save_consultation()
            return self._format_final_response()

        # 2. 添加用户消息到短期记忆 (保持原有代码不变)
        patient_id = self._get_or_create_patient_id()
        self.memory_manager.add_dialogue('patient', message)

        # 添加定期保存逻辑 (保持原有代码不变)
        if hasattr(self, 'memory_manager'):
            self.memory_manager.periodic_save(self.context.turn_count)

        # 3. 更新对话轮次 (保持原有代码不变)
        self.context.turn_count += 1
        logger.info(f"处理消息: {message}, 当前轮次: {self.context.turn_count}, 当前状态: {self.context.state.value}")

        # 【新增 NLU 处理】在初始状态处理前添加
        # 1. 检测用户意图
        intent_result = detect_intent(message, {
            "dialogue": self.memory_manager.short_term.get_current_dialogue(),
            "medical_info": self.context.medical_info
        })
        logger.info(f"意图检测结果: {intent_result.get('primary_intent')}, 置信度: {intent_result.get('confidence')}")

        # 2. 分析上下文关联信息
        context_analysis = self.context_analyzer.analyze_context(
            message,
            {
                "dialogue": self.memory_manager.short_term.get_current_dialogue(),
                "medical_info": self.context.medical_info
            }
        )

        # 3. 如果是紧急情况意图，立即处理
        primary_intent = intent_result.get("primary_intent", "other")
        intent_confidence = intent_result.get("confidence", 0)

        if primary_intent == "emergency" and intent_confidence > 0.7:
            emergency_result = is_emergency_intent(message)
            if emergency_result.get("is_emergency", False) and emergency_result.get("confidence", 0) > 0.7:
                # 设置紧急情况，直接转到转诊流程
                logger.info(f"NLU检测到紧急情况: {emergency_result.get('reason')}")
                self.context.medical_info['emergency_advice'] = emergency_result.get("reason", "检测到紧急情况")
                self.context.medical_info['severity'] = str(emergency_result.get("severity", 8))
                self.context.state = DialogueState.REFERRAL
                self.context.medical_info['referral_urgency'] = "urgent"
                self.current_flow = FLOW_MAPPING[DialogueState.REFERRAL]()
                # 设置LLM流程开关
                if hasattr(self.current_flow, "use_llm_flow"):
                    self.current_flow.use_llm_flow = self.use_llm_flow

        # 4. 初始状态处理 (保持原有逻辑，略微调整)
        if self.context.state == DialogueState.INITIAL:
            logger.info("从初始状态转换到基本信息收集状态")
            # 开始新的问诊会话
            self.memory_manager.start_new_consultation(patient_id)

            self.context.state = DialogueState.COLLECTING_COMBINED_INFO
            self.current_flow = FLOW_MAPPING[DialogueState.COLLECTING_COMBINED_INFO]()
            # 设置LLM流程开关
            if hasattr(self.current_flow, "use_llm_flow"):
                self.current_flow.use_llm_flow = self.use_llm_flow
            logger.info(f"初始化Flow: {self.current_flow.__class__.__name__}")

            # 对于第一次交互，直接返回欢迎问题
            if self.context.turn_count == 1:
                # 添加系统回复到短期记忆
                welcome_msg = "您好，我是您的医疗助手。请问您有什么不舒服的地方吗？"
                self.memory_manager.add_dialogue('doctor', welcome_msg)
                return welcome_msg

            # 5. 从记忆系统检索相关信息 (保持原有代码不变)
            memory_results = self.memory_manager.retrieve_relevant_memory(message, patient_id)
            self._enhance_context_with_memory(memory_results)

        # 【新增 NLU 处理】提取症状和实体
        # 如果是报告症状意图，直接提取症状实体
        if primary_intent == "report_symptom" and self.context.state in [
            DialogueState.COLLECTING_COMBINED_INFO,
            DialogueState.COLLECTING_SYMPTOMS
        ]:
            symptom_result = symptom_entity_recognition(message)
            symptoms = symptom_result.get("symptoms", [])

            # 处理症状实体
            if symptoms:
                logger.info(f"检测到症状实体: {symptoms}")
                # 通过上下文分析器进行交叉引用
                enriched_symptoms = self.context_analyzer.cross_reference_symptoms(
                    symptoms,
                    {
                        "past_symptoms": self.context.medical_info.get("past_symptoms", []),
                        "medical_history": self.context.medical_info.get("medical_history", "")
                    }
                )

                # 添加到当前症状
                for symptom in enriched_symptoms:
                    if isinstance(symptom, dict):
                        self.memory_manager.add_symptom(symptom)
                    else:
                        self.memory_manager.add_symptom({"name": symptom})

                # 更新主要症状字段
                if len(symptoms) > 0 and "main" not in self.context.medical_info:
                    main_symptom = symptoms[0]
                    if isinstance(main_symptom, dict):
                        self.context.medical_info["main"] = main_symptom.get("name", "")
                    else:
                        self.context.medical_info["main"] = main_symptom

        # 检测矛盾信息
        if self.context.turn_count > 1:  # 不是第一次交互
            contradictions = self.context_analyzer.detect_contradiction(message, self.context.medical_info)
            if contradictions.get("has_contradiction", False):
                logger.info(f"检测到矛盾信息: {contradictions.get('contradictions', {})}")
                # 记录矛盾信息
                if "contradictions" not in self.context.medical_info:
                    self.context.medical_info["contradictions"] = {}
                self.context.medical_info["contradictions"].update(contradictions.get("contradictions", {}))

        # 7. 处理消息
        response = None
        if self.current_flow:
            logger.info(
                f"当前Flow: {self.current_flow.__class__.__name__}, use_llm_flow={self.current_flow.use_llm_flow}")

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
                    # 添加系统回复到短期记忆
                    self.memory_manager.add_dialogue('doctor', next_question)
                    return next_question

            # 如果没有下一个问题，才考虑转换状态
            if not next_question:
                self._transition_state()

                # 如果状态已转换，获取新状态的第一个问题
                if self.current_flow:
                    next_question = self.current_flow.get_next_question(self.context)
                    logger.info(f"状态转换后的下一个问题: {next_question}")

            final_response = next_question if next_question else (
                response + "\n 正在生成下一阶段建议，输入任意内容继续" if response else "已收集到您的信息，正在生成下一阶段建议，输入任意内容继续")

            # 6. 将系统回复添加到短期记忆
            self.memory_manager.add_dialogue('doctor', final_response)

            # 7. 检查是否有提到新症状，将其添加到记忆系统
            self._extract_symptoms_from_message(message)

            return final_response

        return "抱歉，当前无法处理您的请求。"

    def _extract_symptoms_from_message(self, message: str):
        """从用户消息中提取症状信息并添加到记忆系统"""
        # 仅在症状收集阶段处理
        if self.context.state not in [DialogueState.COLLECTING_COMBINED_INFO,
                                      DialogueState.COLLECTING_SYMPTOMS]:
            return

        # 使用NLU模块提取症状
        symptom_result = symptom_entity_recognition(message)
        extracted_symptoms = symptom_result.get("symptoms", [])

        # 添加提取的症状到记忆
        for symptom in extracted_symptoms:
            if isinstance(symptom, dict):
                self.memory_manager.add_symptom(symptom)
            else:
                self.memory_manager.add_symptom({'name': symptom})

        # 从医疗信息中提取已存储的症状
        current_symptoms = self.context.medical_info.get('current_symptoms', [])
        main_symptom = self.context.medical_info.get('main')

        # 所有提到的症状
        if current_symptoms:
            for symptom in current_symptoms:
                if isinstance(symptom, dict):
                    self.memory_manager.add_symptom(symptom)
                else:
                    self.memory_manager.add_symptom({'name': symptom})

        # 主要症状
        if main_symptom and isinstance(main_symptom, str):
            self.memory_manager.add_symptom({'name': main_symptom, 'is_main': True})

        # 如果有临时诊断，也添加到记忆系统
        diagnosis = self.context.medical_info.get('temp_diagnosis') or self.context.medical_info.get('diagnosis')
        if diagnosis:
            self.memory_manager.set_temp_diagnosis(diagnosis)

    def set_use_llm_flow(self, use_llm: bool) -> None:
        """设置是否使用LLM驱动的流程"""
        logger.info(f"设置use_llm_flow={use_llm}")
        self.use_llm_flow = use_llm
        # 更新当前Flow的设置
        if self.current_flow and hasattr(self.current_flow, "use_llm_flow"):
            self.current_flow.use_llm_flow = use_llm

    def _handle_contradiction(self, contradictions: Dict[str, Any]):
        """处理检测到的矛盾信息

        Args:
            contradictions: 矛盾信息字典
        """
        if not contradictions or not contradictions.get("has_contradiction", False):
            return

        logger.info(f"处理矛盾信息: {contradictions}")

        # 记录矛盾到医疗信息中
        if "contradictions" not in self.context.medical_info:
            self.context.medical_info["contradictions"] = {}

        # 更新矛盾信息
        self.context.medical_info["contradictions"].update(
            contradictions.get("contradictions", {})
        )

        # 如果当前处于信息收集阶段，可以生成澄清问题
        if self.context.state in [
            DialogueState.COLLECTING_COMBINED_INFO,
            DialogueState.COLLECTING_SYMPTOMS,
            DialogueState.COLLECTING_BASE_INFO
        ]:
            # 这里可以设置一个标志，让当前Flow在下一个问题中询问澄清
            if hasattr(self.current_flow, "needs_clarification"):
                self.current_flow.needs_clarification = True
                self.current_flow.contradiction_fields = list(
                    contradictions.get("contradictions", {}).keys()
                )
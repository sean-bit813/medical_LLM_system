# src/dialogue/flows.py
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
from .states import DialogueState, StateContext, STATE_TRANSITIONS
from .utils import check_emergency
from ..prompts.medical_prompts import MEDICAL_PROMPTS, LLM_FLOW_PROMPTS
from ..app_config import DIALOGUE_CONFIG
from ..llm.api import generate_response, generate_simple_response
from .field_mappings import get_mapping_for_state, format_field_descriptions
from ..config.loader import ConfigLoader


# 设置日志
logger = logging.getLogger(__name__)


class BaseFlow:
    def __init__(self, state: DialogueState):
        self.state = state
        self.prompts = MEDICAL_PROMPTS
        self.required_info = []
        self.current_index = 0
        self.use_llm_flow = True  # 控制是否使用LLM驱动的流程
        # 获取当前状态对应的字段映射
        self.field_mapping = get_mapping_for_state(state.value)


    def format_collected_info(self, context: StateContext) -> str:
        """格式化已收集的医疗信息"""
        formatted = []

        # 使用字段映射增强信息展示
        for key, value in context.medical_info.items():
            field_info = None
            original_key = key

            # 尝试查找字段信息
            if key in self.field_mapping:
                field_info = self.field_mapping[key]
            else:
                # 尝试查找与中文名匹配的字段
                for field_key, info in self.field_mapping.items():
                    if info['zh_name'] == key or info['zh_name'] in key:
                        field_info = info
                        original_key = field_key
                        break

            # 根据字段信息格式化输出
            if field_info:
                formatted.append(f"{original_key}({field_info['zh_name']}): {value}")
            else:
                formatted.append(f"{key}: {value}")

        return "\n".join(formatted) if formatted else "暂无收集的信息"

    def process_response(self, response: str, context: StateContext) -> bool:
        """处理用户回复，返回是否紧急情况"""
        # 只使用LLM处理回复
        return self.process_response_with_llm(response, context)

    def get_next_question(self, context: StateContext) -> Optional[str]:
        """获取下一个问题，从LLM生成"""
        # 首次问诊初始化的特殊处理
        if self.state == DialogueState.COLLECTING_COMBINED_INFO and self.current_index == 0 and not context.medical_info:
            # 对于第一个问题，使用标准欢迎语
            logger.info("首次问诊，返回欢迎语和首个问题")
            return "请问您有什么不舒服的地方吗？"
        # 非信息收集阶段使用固定模板
        non_collection_states = [
            DialogueState.DIAGNOSIS,
            DialogueState.MEDICAL_ADVICE,
            DialogueState.REFERRAL,
            DialogueState.EDUCATION
        ]

        if self.state in non_collection_states:
            # 检查是否有对应的LLM提示词
            if not LLM_FLOW_PROMPTS.get(self.state.value):
                logger.warning(f"状态 {self.state.value} 没有对应的LLM提示词")
                return "请稍等，我正在分析您的情况。"

            # 检查是否有问题模板
            if "next_question_template" not in LLM_FLOW_PROMPTS.get(self.state.value, {}):
                logger.warning(f"状态 {self.state.value} 没有问题模板")
                return "请稍等，我正在分析您的情况。"

            # 使用状态对应的模板生成输出
            prompt_template = LLM_FLOW_PROMPTS[self.state.value]["next_question_template"]
            collected_info = self.format_collected_info(context)

            prompt = prompt_template.format(
                collected_info=collected_info
            )

            logger.info(f"生成分析结果提示词: {prompt[:100]}...")
            result = generate_simple_response(prompt)
            logger.info(f"LLM生成结果: {result}")

            return result

        # 检查是否已收集完所有必要信息
        if self.check_completion_with_llm(context):
            return None

        # 准备提示词
        prompt_template = LLM_FLOW_PROMPTS[self.state.value]["next_question_template"]

        # 收集已有信息
        collected_info = self.format_collected_info(context)

        # 获取字段详细描述
        field_descriptions = format_field_descriptions(self.state.value)

        # 生成可用字段列表 - 排除已收集的字段
        available_fields = []
        for field in self.required_info:
            if field not in context.medical_info:
                if field in self.field_mapping:
                    available_fields.append(f"{field} ({self.field_mapping[field]['zh_name']})")
                else:
                    available_fields.append(field)

        # 将可用字段转换为带编号的列表形式
        field_list = "\n".join([f"{i + 1}. {field}" for i, field in enumerate(available_fields)])

        # 生成提示词
        prompt = prompt_template.format(
            collected_info=collected_info,
            required_info=", ".join(self.required_info),
            field_descriptions=field_descriptions,
            field_list=field_list
        )

        # 调用LLM生成问题及下一个字段
        logger.info(f"生成问题提示词: {prompt[:100]}...")
        result = generate_simple_response(prompt)
        logger.info(f"LLM生成结果: {result}")

        # 解析结果，获取问题和下一个字段
        next_field = self.extract_next_field_from_result(result, available_fields)
        question = self.extract_question_from_result(result)

        # 记录下一个问题的字段
        if next_field:
            # 去除可能包含的中文名称部分
            next_field = next_field.split(" (")[0] if " (" in next_field else next_field
            context.last_question_field = next_field
            logger.info(f"下一个问题字段: {next_field}")

        return question if question else "请告诉我更多关于您的情况"

    def extract_next_field_from_result(self, result: str, available_fields: List[str]) -> Optional[str]:
        """从LLM结果中提取下一个字段"""
        # 寻找显式的字段标识
        if "FIELD:" in result:
            lines = result.split('\n')
            for line in lines:
                if "FIELD:" in line:
                    field_part = line.split("FIELD:")[1].strip()
                    # 检查是否包含编号
                    if field_part and field_part[0].isdigit() and '. ' in field_part:
                        field_part = field_part.split('. ', 1)[1]
                    return field_part

        # 备用方法：尝试找到所有可能的字段匹配
        for field_info in available_fields:
            # 提取原始字段名
            field = field_info.split(" (")[0] if " (" in field_info else field_info
            # 检查原始字段名
            if field in result:
                return field

        # 如果找不到，返回第一个未收集的字段
        if available_fields:
            first_field = available_fields[0]
            return first_field.split(" (")[0] if " (" in first_field else first_field

        return None

    def extract_question_from_result(self, result: str) -> Optional[str]:
        """从LLM结果中提取问题"""
        if "QUESTION:" in result:
            lines = result.split('\n')
            for i, line in enumerate(lines):
                if "QUESTION:" in line:
                    question = line.split("QUESTION:")[1].strip()
                    # 检查是否有多行内容
                    j = i + 1
                    while j < len(lines) and "FIELD:" not in lines[j] and "QUESTION:" not in lines[j]:
                        question += " " + lines[j].strip()
                        j += 1
                    return question

        # 如果没有找到明确的问题标记，返回整个结果作为问题
        # 但需要移除FIELD部分
        if "FIELD:" in result:
            # 尝试移除FIELD行
            lines = result.split('\n')
            question_lines = []
            for line in lines:
                if "FIELD:" not in line:
                    question_lines.append(line)
            return '\n'.join(question_lines).strip()

        return result.strip()

    def process_response_with_llm(self, response: str, context: StateContext) -> bool:
        """使用LLM处理用户回复，提取信息并检测紧急情况"""
        if not response.strip():
            return False

        # 判断当前状态是否需要信息提取
        info_collection_states = [
            DialogueState.COLLECTING_BASE_INFO,
            DialogueState.COLLECTING_SYMPTOMS,
            DialogueState.COLLECTING_COMBINED_INFO,
            DialogueState.LIFE_STYLE
        ]

        # 只有信息收集阶段才进行信息提取
        if self.state in info_collection_states:
            # 获取当前正在询问的字段
            current_field = context.last_question_field if hasattr(context, 'last_question_field') else None
            logger.info(f"处理响应，上一个问题字段: {current_field}")

            # 提取信息
            extracted = self.extract_info_with_llm(response, context, current_field)
            logger.info(f"信息提取结果: extracted={extracted}, current_field={current_field}")

            # 如果没有成功提取任何信息，并且有明确的当前字段，直接将整个回复映射到该字段
            if not extracted and current_field:
                context.medical_info[current_field] = response
                logger.info(f"没有成功提取信息，直接映射整个回复到当前字段: {current_field}={response}")

            # 检测严重程度 (只在症状收集阶段进行)
            if self.state in [DialogueState.COLLECTING_SYMPTOMS, DialogueState.COLLECTING_COMBINED_INFO]:
                severity = self.extract_severity_with_llm(response, context)
                if severity is not None:
                    context.medical_info['severity'] = str(severity)
        else:
            # 非信息收集阶段，只是简单记录用户回复
            logger.info(f"当前阶段 {self.state.value} 不需要信息提取，只记录用户回复")

        # 检查是否需要紧急处理 (只在相关阶段进行)
        if self.state in [DialogueState.COLLECTING_SYMPTOMS, DialogueState.COLLECTING_COMBINED_INFO]:
            return self.check_emergency_with_llm(response, context)

        return False

    def should_transition(self, context: StateContext) -> bool:
        """判断是否应该转换状态"""
        # 使用LLM判断是否完成
        is_complete = self.check_completion_with_llm(context)
        logger.info(f"是否应该转换状态: {is_complete}")
        return is_complete

    def get_next_state(self, context: StateContext) -> Optional[DialogueState]:
        """获取下一个状态"""
        if not self.should_transition(context):
            return self.state

        # 获取可能的下一个状态
        possible_states = STATE_TRANSITIONS.get(self.state, [])
        next_state = possible_states[0] if possible_states else None

        # 记录状态转换
        if next_state:
            logger.info(f"状态将转换为: {next_state.value}")

        self.reset()
        return next_state

    def extract_info_with_llm(self, response: str, context: StateContext, current_field: Optional[str] = None) -> bool:
        """使用LLM从用户回复中提取信息"""
        # 简单问候语直接跳过信息提取
        simple_greetings = ["你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "开始", "start"]
        if response.strip().lower() in simple_greetings and not current_field:
            logger.info(f"检测到简单问候，跳过信息提取: {response}")
            return False

        # 检查是否有对应的LLM提示词
        if not LLM_FLOW_PROMPTS.get(self.state.value):
            logger.warning(f"状态 {self.state.value} 没有对应的LLM提示词")
            return False

        # 准备提示词
        prompt_template = LLM_FLOW_PROMPTS[self.state.value]["info_extraction_template"]

        # 获取字段详细描述
        field_descriptions = format_field_descriptions(self.state.value)

        # 增强提示词，告诉LLM当前正在询问哪个字段
        field_description = ""
        if current_field and current_field in self.field_mapping:
            field_description = f"{current_field}({self.field_mapping[current_field]['zh_name']}): {self.field_mapping[current_field]['description']}"

        # 生成提示词
        prompt = prompt_template.format(
            user_response=response,
            field_descriptions=field_descriptions,
            current_question_field=current_field or "未指定",
            current_field_description=field_description
        )

        # 调用LLM提取信息
        logger.info(f"信息提取提示词: {prompt[:100]}...")
        extraction_result = generate_simple_response(prompt)
        logger.info(f"信息提取结果: {extraction_result}")

        # 解析结果并更新上下文
        extracted_anything = False

        if extraction_result:
            try:
                # 如果LLM回复表示没有提取到信息，直接返回False
                if "未提供" in extraction_result or "未找到" in extraction_result or "无法提取" in extraction_result:
                    return False

                # 尝试解析结果，找到键值对
                lines = extraction_result.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # 只要不是空字符串就保存
                        if value:
                            # 对于中文字段名，尝试映射到英文字段名
                            mapped_key = None

                            # 1. 检查是否有对应的字段映射
                            for field_key, info in self.field_mapping.items():
                                if (info['zh_name'] == key or
                                        info['zh_name'] in key or
                                        field_key.lower() == key.lower() or
                                        field_key in key):
                                    mapped_key = field_key
                                    break

                            # 2. 如果没有找到匹配，但有当前字段，使用当前字段
                            if not mapped_key and current_field:
                                mapped_key = current_field

                            # 3. 如果仍然没有找到匹配，保留原始键
                            if not mapped_key:
                                mapped_key = key

                            # 记录映射信息并保存
                            logger.info(f"字段映射: {key} -> {mapped_key} = {value}")
                            context.medical_info[mapped_key] = value
                            extracted_anything = True

                # 如果没有成功提取任何信息，并且有明确的当前字段，直接将整个回复映射到该字段
                if not extracted_anything and current_field:
                    context.medical_info[current_field] = response
                    logger.info(f"没有成功提取信息，直接映射整个回复到当前字段: {current_field}={response}")
                    extracted_anything = True

            except Exception as e:
                logger.error(f"解析提取信息时出错: {e}")

        return extracted_anything

    def extract_severity_with_llm(self, response: str, context: StateContext) -> Optional[int]:
        """使用LLM从用户回复中评估症状的严重程度（1-10）"""
        # 准备提示词
        prompt_template = LLM_FLOW_PROMPTS.get("severity_assessment_template", """
        基于患者的描述："{user_response}"
        
        以及已收集信息：{info}

        请评估症状的严重程度，使用1-10的数字，其中1代表非常轻微，10代表非常严重。
        仅返回一个数字（1-10）。
        """)

        # 生成提示词
        prompt = prompt_template.format(user_response=response, info= self.format_collected_info(context))

        # 调用LLM评估严重程度
        logger.info(f"严重程度评估提示词: {prompt}")
        severity_result = generate_simple_response(prompt)
        logger.info(f"严重程度评估结果: {severity_result}")

        # 尝试从结果中提取数字
        try:
            # 提取数字
            import re
            numbers = re.findall(r'\b([1-9]|10)\b', severity_result)
            if numbers:
                severity = int(numbers[0])
                return min(max(severity, 1), 10)  # 确保在1-10范围内
        except Exception as e:
            logger.error(f"解析严重程度时出错: {e}")

        return None

    def check_emergency_with_llm(self, response: str, context: StateContext) -> bool:
        """使用LLM检查是否存在紧急情况"""
        # 检查是否可以使用传统方法判断紧急情况
        severity = context.medical_info.get('severity')
        if severity and severity.isdigit() and int(severity) >= 8:
            return True

        # 准备提示词
        prompt_template = LLM_FLOW_PROMPTS.get("emergency_assessment_template", """
        基于患者的描述："{user_response}"
        以及已收集的信息：
        {collected_info}

        评估是否存在需要紧急医疗干预的情况。
        紧急情况包括但不限于：严重疼痛、呼吸困难、胸痛、大出血、意识不清等。

        请仅回答"是"或"否"。
        """)

        # 生成提示词
        prompt = prompt_template.format(
            user_response=response,
            collected_info=self.format_collected_info(context)
        )

        # 调用LLM评估紧急情况
        logger.info(f"紧急情况评估提示词: {prompt}")
        emergency_result = generate_simple_response(prompt)
        logger.info(f"紧急情况评估结果: {emergency_result}")

        # 解析结果
        if emergency_result and any(
                keyword in emergency_result.lower() for keyword in ["是", "紧急", "emergency", "urgent", "yes"]):
            return True

        return False

    def check_completion_with_llm(self, context: StateContext) -> bool:
        """使用LLM检查信息是否已收集完整，严格参考required_info列表"""
        # 检查是否有对应的LLM提示词
        if not LLM_FLOW_PROMPTS.get(self.state.value):
            logger.warning(f"状态 {self.state.value} 没有对应的LLM提示词")
            return False

        # 准备提示词
        prompt_template = LLM_FLOW_PROMPTS[self.state.value]["completion_check_template"]

        # 收集已有信息
        collected_info = self.format_collected_info(context)

        # 准备必须收集的字段列表
        required_fields = []
        for field in self.required_info:
            field_desc = ""
            if field in self.field_mapping:
                field_desc = f"{field} ({self.field_mapping[field]['zh_name']}): {self.field_mapping[field]['description']}"
            else:
                field_desc = field
            required_fields.append(field_desc)

        # 生成提示词
        prompt = prompt_template.format(
            collected_info=collected_info,
            required_fields="\n".join([f"- {field}" for field in required_fields])
        )

        # 调用LLM检查完整性
        logger.info(f"完整性检查提示词: {prompt[:100]}...")

        # 要求简洁明确的回答
        response_prompt = """
            请只回答"完整"或"不完整"。不要解释理由，只给一个词的回答。
            """
        prompt += response_prompt

        completion_result = generate_simple_response(prompt)
        logger.info(f"完整性检查结果: {completion_result}")

        # 解析结果
        if "完整" in completion_result and "不完整" not in completion_result:
            return True

        return False

    def reset(self):
        """重置当前流程"""
        self.current_index = 0


class BaseInfoFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.COLLECTING_BASE_INFO)
        self.required_info = ["age", "gender", "medical_history", "allergy", "medication"]


class SymptomFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.COLLECTING_SYMPTOMS)
        self.required_info = ["main_symptoms", "duration", "severity", "pattern", "factors", "associated"]

    def process_response(self, response: str, context: StateContext) -> bool:
        """处理用户回复，对于症状收集需要特别关注紧急情况"""
        # 使用LLM处理回复
        is_emergency = self.process_response_with_llm(response, context)

        if is_emergency:
            context.medical_info['emergency_advice'] = "LLM检测到紧急情况，建议立即就医"
            context.state = DialogueState.REFERRAL
            context.medical_info['referral_urgency'] = "urgent"
            self.reset()
            return True
        return False

    def get_next_state(self, context: StateContext) -> DialogueState:
        if not self.should_transition(context):
            return self.state
        self.reset()
        return DialogueState.LIFE_STYLE


class CombinedInfoFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.COLLECTING_COMBINED_INFO)
        # 合并基本信息和症状信息的所有字段
        self.required_info = [
            "main", "duration", "severity",  # 症状相关字段（优先）
            "age", "gender",  # 高优先级基本信息
            "pattern", "factors", "associated",  # 症状详细信息
            "medical_history", "allergy", "medication"  # 低优先级基本信息
        ]
        # 添加优先级信息，指导LLM生成问题的顺序
        self.field_priorities = {
            "main": 10,  # 最高优先级：主要症状
            "duration": 9,
            "severity": 8,
            "age": 7,
            "gender": 6,
            "pattern": 5,
            "factors": 4,
            "medical_history": 3,
            "allergy": 2,
            "medication": 1
        }

    def get_next_state(self, context: StateContext) -> DialogueState:
        if not self.should_transition(context):
            return self.state
        self.reset()
        return DialogueState.LIFE_STYLE  # 信息收集完成后直接进入生活习惯收集

class LifeStyleFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.LIFE_STYLE)
        self.required_info = ["sleep", "diet", "exercise", "work", "smoke_drink"]

    def get_next_state(self, context: StateContext) -> DialogueState:
        if not self.should_transition(context):
            return self.state
        self.reset()
        return DialogueState.DIAGNOSIS


class DiagnosisFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.DIAGNOSIS)
        self.required_info = []

    def get_next_state(self, context: StateContext) -> DialogueState:
        # 获取严重程度，无论是硬编码还是LLM评估的
        severity_value = context.medical_info.get("severity", "0")
        try:
            severity = int(severity_value)
        except (ValueError, TypeError):
            # 如果无法转换为整数，假设情况不严重
            severity = 0

        return (DialogueState.REFERRAL if severity >= 5
                else DialogueState.MEDICAL_ADVICE)


class MedicalAdviceFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.MEDICAL_ADVICE)
        self.required_info = []

    def get_next_state(self, context: StateContext) -> DialogueState:
        return DialogueState.EDUCATION


class ReferralFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.REFERRAL)
        self.required_info = []

    def get_next_state(self, context: StateContext) -> DialogueState:
        return DialogueState.EDUCATION


class EducationFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.EDUCATION)
        self.required_info = []  # 教育阶段是单向输出,不需要收集信息

    def get_next_state(self, context: StateContext) -> DialogueState:
        return DialogueState.ENDED


# 更新FLOW_MAPPING - 保持原来的映射不变
FLOW_MAPPING = {
    DialogueState.COLLECTING_COMBINED_INFO: CombinedInfoFlow,
    DialogueState.COLLECTING_BASE_INFO: BaseInfoFlow,
    DialogueState.COLLECTING_SYMPTOMS: SymptomFlow,
    DialogueState.LIFE_STYLE: LifeStyleFlow,
    DialogueState.DIAGNOSIS: DiagnosisFlow,
    DialogueState.MEDICAL_ADVICE: MedicalAdviceFlow,
    DialogueState.REFERRAL: ReferralFlow,
    DialogueState.EDUCATION: EducationFlow
}
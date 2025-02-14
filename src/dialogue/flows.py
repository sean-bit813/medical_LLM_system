# src/dialogue/flows.py
from typing import Dict, Optional, List
from datetime import datetime
from .states import DialogueState, StateContext, STATE_TRANSITIONS
from .utils import check_emergency
from ..prompts.medical_prompts import MEDICAL_PROMPTS
from ..config import DIALOGUE_CONFIG


class BaseFlow:
    def __init__(self, state: DialogueState):
        self.state = state
        self.prompts = MEDICAL_PROMPTS
        self.required_info = []
        self.current_index = 0

    def get_next_question(self, context: StateContext) -> (Optional[str], bool):
        if self.current_index >= len(self.required_info):
            return None
        prompt_key = self.required_info[self.current_index]
        self.current_index += 1
        return self.prompts.get(self.state.value, {}).get(prompt_key)

    def process_response(self, response: str, context: StateContext) -> bool:

        if len(context.medical_info) == 0 and self.current_index == 0:
            return False # not emergency

        if len(self.required_info) == 0:
            return False

        if self.current_index <= len(self.required_info):
            key = self.required_info[self.current_index - 1]
            context.medical_info[key] = response
            # self.current_index += 1
            return False

    def should_transition(self, context: StateContext) -> bool:
        return self.current_index >= len(self.required_info)

    def get_next_state(self, context: StateContext) -> Optional[DialogueState]:
        if not self.should_transition(context):
            return self.state
        possible_states = STATE_TRANSITIONS.get(self.state, [])
        self.reset()
        return possible_states[0] if possible_states else None

    def reset(self):
        self.current_index = 0


class BaseInfoFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.COLLECTING_BASE_INFO)
        self.required_info = ["age", "gender", "medical_history", "allergy", "medication"]


class SymptomFlow(BaseFlow):
    def __init__(self):
        super().__init__(DialogueState.COLLECTING_SYMPTOMS)
        self.required_info = ["main", "duration", "severity", "pattern", "factors", "associated"]

    def process_response(self, response: str, context: StateContext) -> bool:
        super().process_response(response, context)

        # 检查是否紧急
        is_emergency, advice = check_emergency(context.medical_info)
        if is_emergency:
            context.medical_info['emergency_advice'] = advice
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
        severity = int(context.medical_info.get("severity", 0))
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


# 更新FLOW_MAPPING
FLOW_MAPPING = {
    DialogueState.COLLECTING_BASE_INFO: BaseInfoFlow,
    DialogueState.COLLECTING_SYMPTOMS: SymptomFlow,
    DialogueState.LIFE_STYLE: LifeStyleFlow,
    DialogueState.DIAGNOSIS: DiagnosisFlow,
    DialogueState.MEDICAL_ADVICE: MedicalAdviceFlow,
    DialogueState.REFERRAL: ReferralFlow,
    DialogueState.EDUCATION: EducationFlow
}

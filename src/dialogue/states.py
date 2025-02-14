# src/dialogue/states.py
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


class DialogueState(Enum):
    """对话状态枚举"""
    INITIAL = "initial"
    COLLECTING_BASE_INFO = "collecting_base_info"
    COLLECTING_SYMPTOMS = "collecting_symptoms"
    LIFE_STYLE = "life_style"
    DIAGNOSIS = "diagnosis"
    MEDICAL_ADVICE = "medical_advice"
    REFERRAL = "referral"  # 包含紧急和常规转诊
    EDUCATION = "education"
    ENDED = "ended"


@dataclass
class StateContext:
    """状态上下文"""
    state: DialogueState
    user_info: Dict
    medical_info: Dict
    start_time: datetime
    turn_count: int = 0
    last_update: Optional[datetime] = None

    def update(self, **kwargs) -> None:
        """更新上下文"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now()
        self.turn_count += 1


STATE_TRANSITIONS = {
    DialogueState.INITIAL: [DialogueState.COLLECTING_BASE_INFO],
    DialogueState.COLLECTING_BASE_INFO: [DialogueState.COLLECTING_SYMPTOMS],
    DialogueState.COLLECTING_SYMPTOMS: [DialogueState.LIFE_STYLE, DialogueState.REFERRAL],
    DialogueState.LIFE_STYLE: [DialogueState.DIAGNOSIS],
    DialogueState.DIAGNOSIS: [DialogueState.MEDICAL_ADVICE, DialogueState.REFERRAL],
    DialogueState.MEDICAL_ADVICE: [DialogueState.EDUCATION],
    DialogueState.REFERRAL: [DialogueState.EDUCATION],
    DialogueState.EDUCATION: [DialogueState.ENDED],
    DialogueState.ENDED: [DialogueState.ENDED]
}
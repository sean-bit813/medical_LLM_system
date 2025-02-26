from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from ..config.loader import ConfigLoader

# 加载状态配置
states_config = ConfigLoader.load_json_config('states.json')

# 创建DialogueState枚举
DialogueState = Enum('DialogueState', {
    state_name: state_value
    for state_name, state_value in states_config['dialogue_states'].items()
})

# 加载状态转换规则
_raw_transitions = states_config['state_transitions']
STATE_TRANSITIONS = {}

# 将字符串状态转换为枚举对象
for state_from_str, states_to_str in _raw_transitions.items():
    from_state = next((s for s in DialogueState if s.value == state_from_str), None)

    if from_state:
        to_states = []
        for state_to_str in states_to_str:
            to_state = next((s for s in DialogueState if s.value == state_to_str), None)
            if to_state:
                to_states.append(to_state)

        STATE_TRANSITIONS[from_state] = to_states


@dataclass
class StateContext:
    """状态上下文"""
    state: DialogueState
    user_info: Dict
    medical_info: Dict
    start_time: datetime
    turn_count: int = 0
    last_update: Optional[datetime] = None
    last_question_field: Optional[str] = None

    def update(self, **kwargs) -> None:
        """更新上下文"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now()
        self.turn_count += 1


STATE_TRANSITIONS = {
    DialogueState.INITIAL: [DialogueState.COLLECTING_COMBINED_INFO],
    DialogueState.COLLECTING_COMBINED_INFO: [DialogueState.LIFE_STYLE, DialogueState.REFERRAL],
    DialogueState.LIFE_STYLE: [DialogueState.DIAGNOSIS],  # 保持不变
    DialogueState.DIAGNOSIS: [DialogueState.MEDICAL_ADVICE, DialogueState.REFERRAL],  # 保持不变
    DialogueState.MEDICAL_ADVICE: [DialogueState.EDUCATION],  # 保持不变
    DialogueState.REFERRAL: [DialogueState.EDUCATION],  # 保持不变
    DialogueState.EDUCATION: [DialogueState.ENDED],  # 保持不变
    DialogueState.ENDED: [DialogueState.ENDED]  # 保持不变
}
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    MATCH = "MATCH"
    REJECT = "REJECT"
    REQUEST_INFO = "REQUEST_INFO"


class LabRange(BaseModel):
    min_value: float
    max_value: float


class PatientRecord(BaseModel):
    patient_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    labs: Dict[str, float] = Field(default_factory=dict)


class TrialCriteria(BaseModel):
    age_min: int = Field(..., ge=0, le=120)
    age_max: int = Field(..., ge=0, le=120)
    required_conditions: List[str] = Field(default_factory=list)
    excluded_conditions: List[str] = Field(default_factory=list)
    required_medications: List[str] = Field(default_factory=list)
    prohibited_medications: List[str] = Field(default_factory=list)
    lab_ranges: Dict[str, LabRange] = Field(default_factory=dict)

    @field_validator("age_max")
    @classmethod
    def validate_age_range(cls, value: int, info) -> int:
        age_min = info.data.get("age_min")
        if age_min is not None and value < age_min:
            raise ValueError("age_max must be greater than or equal to age_min")
        return value


class HistoryItem(BaseModel):
    step: int
    action: str
    details: Dict[str, str] = Field(default_factory=dict)
    reward_delta: float


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    patient: PatientRecord
    trial_criteria: TrialCriteria
    interaction_history: List[HistoryItem] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    remaining_steps: int


class Action(BaseModel):
    action_type: ActionType
    requested_field: Optional[str] = None
    reasoning: str = ""

    @field_validator("requested_field")
    @classmethod
    def normalize_requested_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    decision_score: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=0.0, le=1.0)
    loop_penalty: float = 0.0
    final_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

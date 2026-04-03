from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .models import ActionType, LabRange, PatientRecord, TrialCriteria


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    patient: PatientRecord
    trial_criteria: TrialCriteria
    expected_decision: ActionType
    required_checks: List[str] = Field(default_factory=list)
    max_steps: int = 6


def get_tasks() -> List[TaskDefinition]:
    return [
        TaskDefinition(
            task_id="task_easy_age_only",
            name="Age-Only Eligibility",
            difficulty="easy",
            description=(
                "Decide if the patient matches a trial with only age boundaries."
            ),
            patient=PatientRecord(
                patient_id="P-001",
                age=45,
                gender="female",
                conditions=["hypertension"],
                medications=["lisinopril"],
                labs={"hemoglobin": 13.1},
            ),
            trial_criteria=TrialCriteria(
                age_min=18,
                age_max=65,
            ),
            expected_decision=ActionType.MATCH,
            required_checks=["age"],
            max_steps=4,
        ),
        TaskDefinition(
            task_id="task_medium_age_condition",
            name="Age + Condition Eligibility",
            difficulty="medium",
            description=(
                "Decide if age and diagnosis history satisfy trial requirements."
            ),
            patient=PatientRecord(
                patient_id="P-002",
                age=59,
                gender="male",
                conditions=["hypertension"],
                medications=["amlodipine"],
                labs={"a1c": 7.1},
            ),
            trial_criteria=TrialCriteria(
                age_min=40,
                age_max=70,
                required_conditions=["type2_diabetes"],
                excluded_conditions=["end_stage_renal_disease"],
            ),
            expected_decision=ActionType.REJECT,
            required_checks=["age", "conditions"],
            max_steps=5,
        ),
        TaskDefinition(
            task_id="task_hard_multifactor",
            name="Multi-Factor Eligibility",
            difficulty="hard",
            description=(
                "Decide using age, conditions, medications, and lab thresholds."
            ),
            patient=PatientRecord(
                patient_id="P-003",
                age=63,
                gender="female",
                conditions=["rheumatoid_arthritis", "chronic_kidney_disease_stage2"],
                medications=["methotrexate", "prednisone"],
                labs={"hemoglobin": 11.8, "creatinine": 1.7, "alt": 32.0},
            ),
            trial_criteria=TrialCriteria(
                age_min=40,
                age_max=70,
                required_conditions=["rheumatoid_arthritis"],
                excluded_conditions=["active_hepatitis"],
                required_medications=["methotrexate"],
                prohibited_medications=["cyclophosphamide"],
                lab_ranges={
                    "hemoglobin": LabRange(min_value=10.0, max_value=15.0),
                    "creatinine": LabRange(min_value=0.6, max_value=1.5),
                    "alt": LabRange(min_value=0.0, max_value=45.0),
                },
            ),
            expected_decision=ActionType.REJECT,
            required_checks=["age", "conditions", "medications", "labs"],
            max_steps=7,
        ),
    ]

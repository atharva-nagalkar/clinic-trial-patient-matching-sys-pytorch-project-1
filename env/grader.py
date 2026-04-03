from __future__ import annotations

from typing import Dict, Iterable, Set

from .models import Action, ActionType, PatientRecord, TrialCriteria
from .tasks import TaskDefinition


_REASONING_KEYWORDS = {
    "age": {"age", "years old", "elderly", "adult"},
    "conditions": {"condition", "diagnosis", "history", "comorbidity"},
    "medications": {"medication", "drug", "therapy", "prescription", "meds"},
    "labs": {"lab", "creatinine", "hemoglobin", "a1c", "alt", "ast", "value"},
}


def _normalize(values: Iterable[str]) -> Set[str]:
    return {value.strip().lower() for value in values}


def evaluate_criteria(patient: PatientRecord, criteria: TrialCriteria) -> Dict[str, bool]:
    age_ok = criteria.age_min <= patient.age <= criteria.age_max

    patient_conditions = _normalize(patient.conditions)
    required_conditions_ok = _normalize(criteria.required_conditions).issubset(patient_conditions)
    excluded_conditions_ok = _normalize(criteria.excluded_conditions).isdisjoint(patient_conditions)
    conditions_ok = required_conditions_ok and excluded_conditions_ok

    patient_meds = _normalize(patient.medications)
    required_meds_ok = _normalize(criteria.required_medications).issubset(patient_meds)
    prohibited_meds_ok = _normalize(criteria.prohibited_medications).isdisjoint(patient_meds)
    medications_ok = required_meds_ok and prohibited_meds_ok

    labs_ok = True
    for lab_name, lab_range in criteria.lab_ranges.items():
        lab_value = patient.labs.get(lab_name)
        if lab_value is None:
            labs_ok = False
            break
        if not (lab_range.min_value <= lab_value <= lab_range.max_value):
            labs_ok = False
            break

    return {
        "age": age_ok,
        "conditions": conditions_ok,
        "medications": medications_ok,
        "labs": labs_ok,
    }


def compute_expected_decision(patient: PatientRecord, criteria: TrialCriteria) -> ActionType:
    checks = evaluate_criteria(patient, criteria)
    return ActionType.MATCH if all(checks.values()) else ActionType.REJECT


def _extract_reasoning_checks(reasoning: str) -> Set[str]:
    text = reasoning.lower()
    covered = set()
    for check_name, keywords in _REASONING_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            covered.add(check_name)
    return covered


def extract_reasoning_checks(reasoning: str) -> Set[str]:
    """Public helper used by environment rewards and final grading logic."""
    return _extract_reasoning_checks(reasoning)


def compute_reasoning_score(
    required_checks: Iterable[str],
    requested_fields: Iterable[str],
    reasoning: str,
) -> float:
    required = [item.lower() for item in required_checks]
    if not required:
        return 1.0

    requested = {field.lower() for field in requested_fields}
    inferred = _extract_reasoning_checks(reasoning)
    covered = requested.union(inferred)

    matched = sum(1 for check in required if check in covered)
    return matched / len(required)


def grade_final_decision(
    task: TaskDefinition,
    action: Action,
    requested_fields: Iterable[str],
) -> Dict[str, float | str]:
    expected_decision = compute_expected_decision(task.patient, task.trial_criteria)
    decision_score = 1.0 if action.action_type == expected_decision else 0.0
    reasoning_score = compute_reasoning_score(
        required_checks=task.required_checks,
        requested_fields=requested_fields,
        reasoning=action.reasoning,
    )

    # Weighted deterministic score with partial credit for reasoning coverage.
    final_score = round((0.6 * decision_score) + (0.4 * reasoning_score), 4)

    return {
        "expected_decision": expected_decision.value,
        "decision_score": decision_score,
        "reasoning_score": reasoning_score,
        "final_score": final_score,
    }

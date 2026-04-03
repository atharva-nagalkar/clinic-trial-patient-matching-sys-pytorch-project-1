from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .grader import compute_reasoning_score, extract_reasoning_checks, grade_final_decision
from .models import Action, ActionType, HistoryItem, Observation, Reward
from .tasks import TaskDefinition, get_tasks


class ClinicalTrialPatientMatchingEnvironment:
    """OpenEnv-style environment for clinical trial patient matching."""

    def __init__(self, tasks: Optional[List[TaskDefinition]] = None):
        self.tasks: List[TaskDefinition] = tasks or get_tasks()
        self.task_index: int = -1
        self.current_task: Optional[TaskDefinition] = None

        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False

        self.history: List[HistoryItem] = []
        self.requested_fields: Set[str] = set()
        self.covered_required_checks: Set[str] = set()

        self.allowed_info_fields = {
            "age",
            "gender",
            "conditions",
            "medications",
            "labs",
        }

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is None:
            self.task_index = (self.task_index + 1) % len(self.tasks)
            self.current_task = self.tasks[self.task_index]
        else:
            matched = [task for task in self.tasks if task.task_id == task_id]
            if not matched:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.current_task = matched[0]

        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.history = []
        self.requested_fields = set()
        self.covered_required_checks = set()

        return self._build_observation()

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")

        if self.done:
            reward = Reward(
                step_reward=0.0,
                cumulative_reward=self.cumulative_reward,
                decision_score=0.0,
                reasoning_score=0.0,
                loop_penalty=0.0,
                final_score=None,
            )
            return self._build_observation(), reward, self.done, {"warning": "Episode already completed."}

        self.step_count += 1
        task = self.current_task

        step_reward = 0.0
        loop_penalty = 0.0
        decision_score = 0.0
        reasoning_score = 0.0
        final_score = None

        info: Dict[str, Any] = {
            "task_id": task.task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "step": self.step_count,
        }

        try:
            action_model = action if isinstance(action, Action) else Action(**action)
        except Exception:
            # Invalid action payload gets deterministic penalty instead of crashing the episode.
            invalid_penalty = -0.2
            step_reward += invalid_penalty
            info["message"] = "Invalid action payload."
            info["invalid_action"] = True

            self.cumulative_reward = round(self.cumulative_reward + step_reward, 4)
            self.history.append(
                HistoryItem(
                    step=self.step_count,
                    action="INVALID",
                    details={"requested_field": "", "reasoning": ""},
                    reward_delta=round(step_reward, 4),
                )
            )

            reward = Reward(
                step_reward=round(step_reward, 4),
                cumulative_reward=self.cumulative_reward,
                decision_score=0.0,
                reasoning_score=0.0,
                loop_penalty=0.0,
                final_score=None,
            )
            return self._build_observation(), reward, self.done, info

        required_checks = {check.lower() for check in task.required_checks}
        reasoning_hits = extract_reasoning_checks(action_model.reasoning)
        relevant_reasoning_hits = reasoning_hits.intersection(required_checks)

        # Reward meaningful explanation when it references task-relevant checks.
        if relevant_reasoning_hits:
            step_reward += 0.1
            info["reasoning_reward"] = 0.1

        if action_model.action_type == ActionType.REQUEST_INFO:
            field = action_model.requested_field
            if field is None:
                step_reward -= 0.2
                info["message"] = "Missing requested_field for REQUEST_INFO action."
                info["invalid_action"] = True
            elif field in self.requested_fields:
                loop_penalty = -0.2
                step_reward += loop_penalty
                info["message"] = f"Repeated info request for '{field}'."
                info["loop_detected"] = True
            elif field not in self.allowed_info_fields:
                step_reward -= 0.2
                info["message"] = f"Unknown requested_field '{field}'."
                info["invalid_action"] = True
            else:
                self.requested_fields.add(field)
                if field in task.required_checks:
                    step_reward += 0.1
                    info["message"] = f"Helpful reasoning step: requested '{field}'."
                else:
                    step_reward -= 0.1
                    info["message"] = f"Irrelevant request for '{field}'."

            # Reward progress when the agent covers a required check not seen before.
            newly_covered = set()
            if field in required_checks:
                newly_covered.add(field)
            newly_covered.update(relevant_reasoning_hits)
            newly_covered.difference_update(self.covered_required_checks)

            if newly_covered:
                step_reward += 0.1
                self.covered_required_checks.update(newly_covered)
                info["progress_reward"] = 0.1
                info["newly_covered_checks"] = sorted(newly_covered)

            # Penalize overlong trajectories that avoid committing to MATCH/REJECT.
            if self.step_count > max(1, len(required_checks)):
                step_reward -= 0.1
                info["delay_penalty"] = -0.1

            if self.step_count >= task.max_steps:
                self.done = True
                timeout_penalty = -0.1
                step_reward += timeout_penalty
                reasoning_score = compute_reasoning_score(
                    required_checks=task.required_checks,
                    requested_fields=self.requested_fields,
                    reasoning=action_model.reasoning,
                )
                final_score = round(max(0.0, 0.4 * reasoning_score), 4)
                info["message"] = "Max steps reached without final decision."
                info["timeout"] = True
                info["expected_decision"] = task.expected_decision.value
                info["decision_score"] = 0.0
                info["reasoning_score"] = reasoning_score
                info["final_score"] = final_score

        else:
            self.done = True
            grading = grade_final_decision(task, action_model, self.requested_fields)
            decision_score = float(grading["decision_score"])
            reasoning_score = float(grading["reasoning_score"])
            final_score = float(grading["final_score"])

            decision_reward = 0.6 if decision_score == 1.0 else -0.6
            reasoning_reward = 0.4 * reasoning_score
            step_reward += decision_reward + reasoning_reward

            info.update(grading)
            info["agent_decision"] = action_model.action_type.value

        self.cumulative_reward = round(self.cumulative_reward + step_reward, 4)

        self.history.append(
            HistoryItem(
                step=self.step_count,
                action=action_model.action_type.value,
                details={
                    "requested_field": action_model.requested_field or "",
                    "reasoning": action_model.reasoning,
                },
                reward_delta=round(step_reward, 4),
            )
        )

        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=self.cumulative_reward,
            decision_score=decision_score,
            reasoning_score=reasoning_score,
            loop_penalty=loop_penalty,
            final_score=final_score,
        )

        return self._build_observation(), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        if self.current_task is None:
            return {
                "task_id": None,
                "step_count": 0,
                "cumulative_reward": 0.0,
                "done": False,
            }

        return {
            "task_id": self.current_task.task_id,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "covered_required_checks": sorted(self.covered_required_checks),
            "requested_fields": sorted(self.requested_fields),
            "history_length": len(self.history),
        }

    def _build_observation(self) -> Observation:
        if self.current_task is None:
            raise RuntimeError("No active task. Call reset() first.")

        remaining_steps = max(0, self.current_task.max_steps - self.step_count)
        return Observation(
            task_id=self.current_task.task_id,
            task_name=self.current_task.name,
            difficulty=self.current_task.difficulty,
            patient=self.current_task.patient,
            trial_criteria=self.current_task.trial_criteria,
            interaction_history=self.history,
            available_actions=[action.value for action in ActionType],
            remaining_steps=remaining_steps,
        )

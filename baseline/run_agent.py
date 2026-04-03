from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List

from openai import OpenAI

from env.environment import ClinicalTrialPatientMatchingEnvironment
from env.grader import evaluate_criteria
from env.models import Action, ActionType, Observation
from env.tasks import get_tasks


SYSTEM_PROMPT = """You are a clinical trial matching agent.
Use the provided patient record and trial criteria.
Return ONLY valid JSON with this schema:
{
  \"action_type\": \"MATCH\" | \"REJECT\" | \"REQUEST_INFO\",
  \"requested_field\": \"age\" | \"gender\" | \"conditions\" | \"medications\" | \"labs\" | null,
  \"reasoning\": \"short rationale\"
}
Prefer making a final decision when enough information is available.
"""


def _build_user_prompt(observation: Observation) -> str:
    payload = observation.model_dump()
    return (
        "Evaluate this clinical trial matching task.\n"
        f"Observation JSON:\n{json.dumps(payload, indent=2)}\n\n"
        "Output only JSON, no markdown."
    )


def make_openai_policy(model: str) -> Callable[[Observation], Action]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for policy=openai")

    client = OpenAI(api_key=api_key)

    def _policy(observation: Observation) -> Action:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation)},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            return Action(**parsed)
        except Exception:
            # Conservative deterministic fallback.
            return Action(
                action_type=ActionType.REJECT,
                requested_field=None,
                reasoning="Invalid model output format; defaulting to reject.",
            )

    return _policy


def rules_policy(observation: Observation) -> Action:
    checks = evaluate_criteria(observation.patient, observation.trial_criteria)
    eligible = all(checks.values())
    decision = ActionType.MATCH if eligible else ActionType.REJECT

    reasoning = (
        f"Age check={checks['age']}; "
        f"Condition check={checks['conditions']}; "
        f"Medication check={checks['medications']}; "
        f"Lab check={checks['labs']}."
    )

    return Action(action_type=decision, requested_field=None, reasoning=reasoning)


def run_tasks(policy_fn: Callable[[Observation], Action], max_steps: int) -> List[Dict[str, object]]:
    env = ClinicalTrialPatientMatchingEnvironment()
    results: List[Dict[str, object]] = []

    for task in get_tasks():
        observation = env.reset(task.task_id)
        done = False
        last_reward = None
        last_info: Dict[str, object] = {}

        steps = 0
        while not done and steps < max_steps:
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            last_reward = reward
            last_info = info
            steps += 1

        if not done:
            # Ensure an episode always terminates with a deterministic final action.
            observation, reward, done, info = env.step(
                Action(
                    action_type=ActionType.REJECT,
                    reasoning="Max policy steps reached; applying safety reject.",
                )
            )
            last_reward = reward
            last_info = info

        final_score = float(last_info.get("final_score", getattr(last_reward, "final_score", 0.0) or 0.0))
        cumulative_reward = float(getattr(last_reward, "cumulative_reward", 0.0))

        results.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "expected_decision": task.expected_decision.value,
                "agent_decision": last_info.get("agent_decision", "N/A"),
                "decision_score": float(last_info.get("decision_score", 0.0)),
                "reasoning_score": float(last_info.get("reasoning_score", 0.0)),
                "final_score": final_score,
                "cumulative_reward": cumulative_reward,
                "steps": steps,
            }
        )

    return results


def print_results(results: List[Dict[str, object]]) -> None:
    print("\n=== Clinical Trial Matching Baseline Results ===")
    for item in results:
        print(
            "- "
            f"{item['task_id']} ({item['difficulty']}): "
            f"decision={item['agent_decision']} | "
            f"score={item['final_score']:.4f} | "
            f"cum_reward={item['cumulative_reward']:.4f} | "
            f"steps={item['steps']}"
        )

    average_score = sum(float(item["final_score"]) for item in results) / len(results)
    average_reward = sum(float(item["cumulative_reward"]) for item in results) / len(results)

    print("\nSummary")
    print(f"- Average final score: {average_score:.4f}")
    print(f"- Average cumulative reward: {average_reward:.4f}")
    print("- Deterministic settings: fixed tasks + temperature=0 + seed=42")


def interactive_demo() -> None:
    env = ClinicalTrialPatientMatchingEnvironment()
    observation = env.reset("task_hard_multifactor")

    print("Interactive Clinical Trial Demo")
    print("Type one of: MATCH, REJECT, REQUEST_INFO")

    done = False
    while not done:
        print("\nCurrent Observation:")
        print(json.dumps(observation.model_dump(), indent=2))

        action_type_raw = input("Action type: ").strip().upper()
        requested_field = input("Requested field (optional): ").strip().lower() or None
        reasoning = input("Reasoning: ").strip()

        try:
            action = Action(
                action_type=ActionType(action_type_raw),
                requested_field=requested_field,
                reasoning=reasoning,
            )
        except Exception as exc:
            print(f"Invalid action: {exc}")
            continue

        observation, reward, done, info = env.step(action)
        print("\nStep result:")
        print(json.dumps({"reward": reward.model_dump(), "info": info}, indent=2))

    print("\nEpisode complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agent on clinical trial matching environment")
    parser.add_argument("--policy", choices=["openai", "rules"], default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--demo", action="store_true", help="Run interactive CLI demo")
    args = parser.parse_args()

    if args.demo:
        interactive_demo()
        return

    if args.policy == "openai":
        policy_fn = make_openai_policy(args.model)
    else:
        policy_fn = rules_policy

    results = run_tasks(policy_fn=policy_fn, max_steps=args.max_steps)
    print_results(results)


if __name__ == "__main__":
    main()

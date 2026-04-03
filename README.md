# Clinical Trial Patient Matching Environment

## 🎯 Agent Objective

The agent must evaluate whether a synthetic patient qualifies for a clinical trial by comparing structured patient data with explicit trial eligibility criteria.
At each step, it should choose `MATCH`, `REJECT`, or `REQUEST_INFO` and provide concise reasoning that reflects the decision process.

A complete, self-contained OpenEnv-style hackathon project that simulates a real clinical workflow: determining whether a patient is eligible for a clinical trial.

This environment is intentionally synthetic and offline-friendly:
- No external medical APIs
- No real patient data
- Deterministic grading and rewards

## Project Overview

Doctors and trial coordinators often evaluate patient eligibility by checking structured criteria such as:
- Age limits
- Required and excluded diagnoses
- Medication constraints
- Lab value thresholds

This environment simulates that workflow and trains/evaluates an AI agent that must:
1. Read a structured patient record
2. Compare against trial criteria
3. Choose one action: MATCH, REJECT, or REQUEST_INFO
4. Provide reasoning

## 🔍 Example Interaction

Simple eligibility scenario (non-technical view):

- Patient data:
  - Age: 63
  - Conditions: rheumatoid_arthritis, chronic_kidney_disease_stage2
  - Medications: methotrexate, prednisone
  - Labs: creatinine = 1.7
- Trial criteria:
  - Age range: 40-70
  - Required condition: rheumatoid_arthritis
  - Required medication: methotrexate
  - Creatinine range: 0.6-1.5
- Agent decision: `REJECT`
- Short reasoning: "The patient meets age and diagnosis requirements, but creatinine is above the trial limit, so the patient is not eligible."

## Motivation

Clinical trial screening is repetitive, high-stakes, and rule-driven.
This project demonstrates how to model that workflow as an interactive environment suitable for:
- Agent training
- Benchmarking reasoning quality
- Hackathon demos

## 🚀 Why This Environment Matters

- Real-world importance: Trial matching directly affects enrollment speed, patient access to treatment, and research quality.
- Beyond simple classification: The agent is evaluated on multi-step decision behavior, information requests, and reasoning quality, not only on a final label.
- Better reasoning evaluation: Deterministic tasks and partial-credit scoring make it clear whether an agent is correct for the right reasons.

## Folder Structure

```text
project/
│
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   ├── grader.py
│
├── baseline/
│   ├── run_agent.py
│
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
```

## How The Environment Works

The runtime class is `ClinicalTrialPatientMatchingEnvironment` and implements:
- `reset(task_id=None)`
- `step(action)`
- `state()`

The `step(action)` return contract is:
- `(observation, reward, done, info)`

### Observation Space

Observation is structured JSON represented by a Pydantic model and includes:
- `task_id`, `task_name`, `difficulty`
- `patient`
  - `age`, `gender`, `conditions`, `medications`, `labs`
- `trial_criteria`
  - age bounds, condition constraints, medication constraints, lab ranges
- `interaction_history`
- `available_actions`
- `remaining_steps`

### Action Space

Supported actions:
- `MATCH`
- `REJECT`
- `REQUEST_INFO`

Action payload fields:
- `action_type` (required)
- `requested_field` (optional, for REQUEST_INFO)
- `reasoning` (optional string)

### Reward Model

The environment uses granular, trajectory-based rewards so agent behavior is shaped throughout the episode, not only at the final decision.

Step-level positive rewards:
- `+0.1` for `REQUEST_INFO` on a required field
- `+0.1` for meaningful reasoning contribution tied to required checks
- `+0.1` for progress toward completing required checks

Final decision reward:
- `+1.0` for a correct final decision

Penalties:
- `-0.2` for repeated `REQUEST_INFO` on the same field
- `-0.1` for irrelevant or unnecessary requests
- `-0.2` for invalid actions
- `-0.1` for excessive steps without a final decision

These step rewards guide behavior during the episode, while the final score evaluates overall task performance.

The final deterministic grader score is in `[0.0, 1.0]`:

$$
\text{final\_score} = 0.6 \cdot \text{decision\_score} + 0.4 \cdot \text{reasoning\_score}
$$

Where:
- `decision_score` is `1.0` if decision matches ground truth else `0.0`
- `reasoning_score` is partial credit based on coverage of required checks

## Tasks

The environment ships with 3 tasks of increasing difficulty.

1. Easy: Age-only eligibility
- Checks: `age`
- Expected decision: `MATCH`

2. Medium: Age + conditions
- Checks: `age`, `conditions`
- Expected decision: `REJECT`

3. Hard: Age + conditions + medications + labs
- Checks: `age`, `conditions`, `medications`, `labs`
- Expected decision: `REJECT`

Each task includes:
- Patient record
- Trial criteria
- Expected correct decision
- Programmatic deterministic grading via `env/grader.py`

## ⚡ Quick Demo

Run the interactive demo:

```bash
python baseline/run_agent.py --demo
```

The demo launches a CLI episode where you can manually choose `MATCH`, `REJECT`, or `REQUEST_INFO`, observe reward updates, and see how reasoning affects final scoring.

## Setup Instructions

From the `project/` folder:

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Run deterministic local baseline (no API key needed):

```bash
python baseline/run_agent.py --policy rules
```

Run OpenAI baseline:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
python baseline/run_agent.py --policy openai --model gpt-4.1-mini
```

Optional interactive CLI demo:

```bash
python baseline/run_agent.py --demo
```

## Docker Usage

From the `project/` folder:

```bash
docker build -t openenv-clinical .
docker run openenv-clinical
```

To run with OpenAI policy in Docker:

```bash
docker run -e OPENAI_API_KEY=your_api_key_here openenv-clinical \
  python baseline/run_agent.py --policy openai --model gpt-4.1-mini
```

## Hugging Face Space Deployment

This project is lightweight and deployable as a Docker Space.

Recommended setup:
1. Create a new Hugging Face Space with SDK set to `Docker`
2. Upload this `project/` directory as the repository root
3. The default container command runs a deterministic CLI baseline

Demo options:
- CLI mode: `python baseline/run_agent.py --demo`
- Automated run: default `CMD` in Dockerfile

No heavy dependencies are required.

## Baseline Results

Deterministic rules baseline (`--policy rules`) sample outcome:
- Easy task final score: `1.0000`
- Medium task final score: `1.0000`
- Hard task final score: `1.0000`
- Average final score: `1.0000`

OpenAI baseline (`--policy openai`) also uses deterministic request settings:
- `temperature=0`
- `seed=42`
- fixed task ordering

## OpenEnv Spec Notes

The `openenv.yaml` file includes:
- Environment name and description
- Action schema
- Observation schema
- Task metadata and expected outcomes

## Production Readiness Notes

- Clean modular separation (models, tasks, grading, environment, baseline)
- Strong typing and validation with Pydantic
- Deterministic scoring for reliable benchmarking
- Safe synthetic data only
- Dockerized for reproducible execution

## 🏁 Summary

This project demonstrates how complex, real-world decision workflows can be modeled as structured environments for training and evaluating next-generation AI agents with interpretable reasoning.
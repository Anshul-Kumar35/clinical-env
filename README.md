---
title: Clinical Trial Optimizer
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - medical-ai
  - llm-agent
  - fastapi
  - docker
---

# AI Clinical Trial Optimizer

**A Meta PyTorch OpenEnv Hackathon Submission**

## 🧬 Environment Description & Motivation
The AI Clinical Trial Optimizer is a lightweight, strictly typed reinforcement learning environment that simulates a critical real-world medical decision process. 

Instead of standard grid-world navigation, this environment tackles a genuine clinical optimization problem: dynamically adjusting treatment dosage to maximize patient recovery while strictly avoiding severe side effects (toxicity). It forces the AI agent to balance efficacy against safety across varying patient cohorts, modeling a real-world trade-off with a programmatic, deterministic grader.

## ⚙️ The State & Action Spaces

### Observation Space (Pydantic Typed)
The environment returns a strict JSON-serializable state containing:
* `level`: The current task difficulty (easy, medium, hard).
* `step_count` & `max_steps`: Trial duration metrics.
* `dosage`: The current treatment intensity (0.0 to 1.0).
* `patients`: A list of patient states tracking `severity` (base resistance), `response` (recovery progress), `recovered` (boolean), and `side_effect` (boolean).
* `trial_active`: Boolean indicating if the episode is ongoing.

### Action Space (Discrete)
The agent can choose one of five strict string actions:
1. `increase_dosage`: Raises dosage by 15% (slight reward penalty).
2. `decrease_dosage`: Lowers dosage by 15% (slight reward penalty).
3. `hold_dosage`: Maintains current dosage.
4. `run_bloodwork`: Checks if current dosage exceeds the cohort's toxicity threshold (costs a slight reward penalty but provides critical safety info).
5. `stop_trial`: Prematurely ends the episode.

## 🎯 Tasks & Difficulty Levels
The environment features 3 mechanically distinct tasks, each evaluated by a deterministic 0.0 - 1.0 grader upon completion.

1. **Easy (Standard Cohort):** 3 patients, standard noise, high side-effect threshold (0.85). Tests basic optimization.
2. **Medium (Vulnerable Cohort):** 5 patients, lower side-effect threshold (0.65). Forces the agent to act conservatively and actively use the `decrease_dosage` action.
3. **Hard (Delayed Efficacy):** 8 patients, high noise, and `delay_effect: True`. The biological response to the drug lags by one step, forcing the LLM to reason temporally rather than just reacting to the immediate previous state.

## 📊 Reward Function & Grader
* **Step-wise Reward:** Provides immediate, continuous signal. The agent receives fractional positive rewards for partial patient response, massive positive rewards (+1.0) for full recovery, and massive penalties (-0.50) if a patient suffers a side effect.
* **Final Grader:** When `done == True`, the environment calculates a strict 0.0 to 1.0 score based on `(Recoveries / Total Patients) - (Side Effects * 0.5)`.

## 🚀 Setup and Usage

### 1. Run the Live Web Server (Phase 1 Validation)
This environment is fully containerized and runs a FastAPI server to comply with OpenEnv automated testing.
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server locally
uvicorn server:app --host 0.0.0.0 --port 7860

# 2. Run with Docker

docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env

# 3. Run the Baseline Inference Script

export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your_api_key_here"

python inference.py

## 📈 Baseline Scores (seed=42, gpt-3.5-turbo, temperature=0.0)

| Task   | Grader Score | Notes                         |
|--------|-------------|--------------------------------|
| Easy   | 1.00        | Standard cohort, 3 patients    |
| Medium | 1.00        | Vulnerable cohort, 5 patients  |
| Hard   | 1.00        | Delayed efficacy, 8 patients   |
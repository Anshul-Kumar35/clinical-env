import random
from typing import Dict, Any, Tuple
from pydantic import BaseModel

# --- OpenEnv Spec: Typed Models ---
class PatientState(BaseModel):
    severity: float
    response: float
    recovered: bool
    side_effect: bool

class Observation(BaseModel):
    level: str
    step_count: int
    max_steps: int
    dosage: float
    patients: list[PatientState]
    trial_active: bool

class Action(BaseModel):
    action: str

class Reward(BaseModel):
    value: float

class ClinicalTrialEnv:
    # Added run_bloodwork
    ACTIONS = ["increase_dosage", "decrease_dosage", "hold_dosage", "stop_trial", "run_bloodwork"]
    
    LEVEL_CONFIG = {
        "easy": {"patients": 3, "noise": 0.02, "max_steps": 8, "side_effect_threshold": 0.85, "delay_effect": False},
        "medium": {"patients": 5, "noise": 0.05, "max_steps": 10, "side_effect_threshold": 0.65, "delay_effect": False},
        "hard": {"patients": 8, "noise": 0.10, "max_steps": 12, "side_effect_threshold": 0.80, "delay_effect": True},
    }

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)
        self.reset("easy")

    def reset(self, level: str = "easy") -> Observation:
        if level not in self.LEVEL_CONFIG:
            raise ValueError(f"Unknown level: {level}")
        
        self.cfg = self.LEVEL_CONFIG[level]
        self.level = level
        self.step_count = 0
        self.max_steps = self.cfg["max_steps"]
        self.noise = self.cfg["noise"]
        self.dosage = 0.50
        self.done = False
        self.patients = []
        self.pending_efficacy = 0.0
        
        for _ in range(self.cfg["patients"]):
            severity = round(self.random.uniform(0.30, 0.85), 2)
            self.patients.append({
                "severity": severity,
                "response": 0.0,
                "recovered": False,
                "side_effect": False,
            })
            
        return self.state()

    def state(self) -> Observation:
        patient_states = [PatientState(**p) for p in self.patients]
        return Observation(
            level=self.level,
            step_count=self.step_count,
            max_steps=self.max_steps,
            dosage=round(self.dosage, 2),
            patients=patient_states,
            trial_active=not self.done
        )

    def _calculate_grader_score(self) -> float:
        if not self.done:
            return 0.0
        total_patients = len(self.patients)
        recoveries = sum(1 for p in self.patients if p["recovered"])
        side_effects = sum(1 for p in self.patients if p["side_effect"])
        score = (recoveries / total_patients) - (side_effects * 0.5)
        return max(0.0, min(1.0, score))

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action. Expected one of {self.ACTIONS}")

        reward = 0.0
        info = {"action": action, "new_recoveries": 0, "new_side_effects": 0, "bloodwork_results": None}

        # Handle actions
        if action == "increase_dosage":
            self.dosage += 0.15
            reward -= 0.02
        elif action == "decrease_dosage":
            self.dosage -= 0.15
            reward -= 0.01
        elif action == "run_bloodwork":
            # New action: Gives insight but costs a tiny bit of reward
            reward -= 0.05
            info["bloodwork_results"] = "Toxicity levels nominal." if self.dosage < self.cfg["side_effect_threshold"] else "Warning: Toxicity critical."
        elif action == "stop_trial":
            self.done = True

        self.dosage = max(0.0, min(1.0, self.dosage))

        current_efficacy = self.dosage
        if self.cfg["delay_effect"]:
            applied_efficacy = self.pending_efficacy
            self.pending_efficacy = current_efficacy
        else:
            applied_efficacy = current_efficacy

        for patient in self.patients:
            if patient["recovered"] or patient["side_effect"]:
                continue

            if self.dosage > self.cfg["side_effect_threshold"]:
                patient["side_effect"] = True
                reward -= 0.50
                info["new_side_effects"] += 1
                continue

            effect = applied_efficacy * (1.0 - patient["severity"])
            effect += self.random.uniform(-self.noise, self.noise)
            patient["response"] += max(0.0, effect)

            if patient["response"] >= 1.00:
                patient["recovered"] = True
                reward += 1.00
                info["new_recoveries"] += 1
            else:
                reward += 0.10 * effect

        self.step_count += 1

        if all(p["recovered"] or p["side_effect"] for p in self.patients):
            self.done = True
        if self.step_count >= self.max_steps:
            self.done = True

        if self.done:
            info["final_grader_score"] = self._calculate_grader_score()

        return self.state(), round(reward, 3), self.done, info
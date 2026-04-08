import os
import json
from openai import OpenAI
from env import ClinicalTrialEnv

def run_baseline(level: str = "easy"):
    # Correct Hackathon environment variables
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    hf_token = os.environ.get("HF_TOKEN")
    
    # HF_TOKEN is often passed in place of the api_key for standard models on Hugging Face infrastructure
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token if hf_token else os.environ.get("OPENAI_API_KEY", "dummy-key")
    )
    
    env = ClinicalTrialEnv(seed=42)
    
    print(f"[START] level={level}")
    observation = env.reset(level)
    done = False
    total_reward = 0.0
    
    while not done:
        obs_dict = observation.model_dump()
        prompt = f"""
        You are an AI clinical trial agent.
        State: {json.dumps(obs_dict)}
        Valid Actions: {env.ACTIONS}
        Output exactly one action string.
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            raw_action = response.choices[0].message.content.strip()
            action = raw_action if raw_action in env.ACTIONS else "hold_dosage"
        except Exception as e:
            action = "hold_dosage"

        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        # Exact logging requirement for the parser
        print(f"[STEP] action={action} reward={reward} done={done}")

    score = info.get('final_grader_score', 0.0)
    print(f"[END] total_reward={round(total_reward, 3)} score={score}")

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")
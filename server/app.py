import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import ClinicalTrialEnv, Action

app = FastAPI()
env = ClinicalTrialEnv()

app = FastAPI(title="Clinical Trial Env API")
env_instance = ClinicalTrialEnv(seed=42)

class ResetRequest(BaseModel):
    level: str = "easy"

@app.post("/reset")
def reset_env(req: ResetRequest = None):
    level = req.level if req else "easy"
    try:
        obs = env_instance.reset(level)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action_req: Action):
    try:
        obs, reward, done, info = env_instance.step(action_req.action)
        return {
            "observation": obs.model_dump(),
            "reward": {"value": reward},
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    return env_instance.state().model_dump()

@app.get("/")
def health():
    return {"status": "ok", "env": "clinical-trial-optimizer"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
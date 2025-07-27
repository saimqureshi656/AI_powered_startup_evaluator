from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loading import generate_response, get_model
from app.prompt import build_question_prompt, build_evaluation_prompt

app = FastAPI()

# Input schema for startup info
class StartupInfo(BaseModel):
    name: str
    industry: str
    pitch: str
    year: str
    funding: str

# Input schema for evaluation request
class EvaluationRequest(BaseModel):
    startup_info: StartupInfo
    founder_answers: list[str]  # Expecting 10 founder answers

# 🔄 Load and use the question-generation model on demand
@app.post("/generate-questions")
def generate_questions(info: StartupInfo):
    prompt = build_question_prompt(info.dict())
    model = get_model("saimqureshi656/startups-question-generation")
    result = generate_response(model, prompt)
    return {"questions": result}

# 🔄 Load and use the evaluation model on demand
@app.post("/evaluate-startup")
def evaluate_startup(req: EvaluationRequest):
    prompt = build_evaluation_prompt(req.startup_info.dict(), req.founder_answers)
    model = get_model("saimqureshi656/llama3-8b-startup-evaluator-lora")
    result = generate_response(model, prompt)
    return {"evaluation": result}

from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loading import generate_response, question_model, evaluation_model
from app.prompt import build_question_prompt, build_evaluation_prompt

app = FastAPI()

class StartupInfo(BaseModel):
    name: str
    industry: str
    pitch: str
    year: str
    funding: str

class EvaluationRequest(BaseModel):
    startup_info: StartupInfo
    founder_answers: list[str]  # list of 10 answers

@app.post("/generate-questions")
def generate_questions(info: StartupInfo):
    prompt = build_question_prompt(info.dict())
    result = generate_response(question_model, prompt)
    return {"questions": result}

@app.post("/evaluate-startup")
def evaluate_startup(req: EvaluationRequest):
    prompt = build_evaluation_prompt(req.startup_info.dict(), req.founder_answers)
    result = generate_response(evaluation_model, prompt)
    return {"evaluation": result}

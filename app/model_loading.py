from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# Hugging Face login
login(token=hf_token)  # Replace with your token

base_model = "meta-llama/Meta-Llama-3-8B"
question_model_id = "saimqureshi656/startups-question-generation"
evaluation_model_id = "saimqureshi656/llama3-8b-startup-evaluator-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def load_lora_model(lora_model_id):
    print(f"🔄 Loading model: {lora_model_id} on {device.upper()}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, lora_model_id)
    model.eval()
    return model

question_model = load_lora_model(question_model_id)
evaluation_model = load_lora_model(evaluation_model_id)

def generate_response(model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

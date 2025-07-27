from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load Hugging Face token from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Constants
base_model = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Shared tokenizer (same for both LoRA models)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# BitsAndBytes config for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Simple in-memory cache for loaded models
loaded_model_cache = {}

def load_lora_model(lora_model_id: str):
    print(f"🔄 Loading model: {lora_model_id} on {device.upper()}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, lora_model_id)
    model.eval()
    return model

def get_model(lora_model_id: str):
    if lora_model_id in loaded_model_cache:
        return loaded_model_cache[lora_model_id]

    # Clear GPU memory before loading a new model
    torch.cuda.empty_cache()
    model = load_lora_model(lora_model_id)
    loaded_model_cache[lora_model_id] = model
    return model

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

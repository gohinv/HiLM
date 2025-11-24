import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16):
    # load .env
    # load_dotenv()
    # hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") # create .env file with your own token
    # login(hf_token)

    # Load the SLM (e.g. Llama 3.2 1B Instruct)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token or "</s>"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,       # or bfloat16/8bit/4bit for on-device
        device_map="auto"
    )

    return model, tokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading local LLM ({MODEL_ID})... This may take a minute.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Initialize pipeline as 'pipe'
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
)

def generate_response(system_prompt: str, user_prompt: str) -> str:
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    result = pipe(
        full_prompt, 
        do_sample=True, 
        temperature=0.7,
        max_new_tokens=150,
        max_length=None,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        clean_up_tokenization_spaces=True
    )
    
    return result[0]["generated_text"].strip()

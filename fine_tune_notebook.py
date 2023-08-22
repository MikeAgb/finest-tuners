import os
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from peft import (
    PeftConfig,
    TaskType,
    get_peft_model,
    PeftModel
)
import torch
import transformers

# Hugging Face login
access_token = "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"
login(token=access_token)

# Test dataset creation
data = {
    "full_prompt": ['hello there']
}
df = pd.DataFrame(data)
df.to_csv('test.csv', index=False)


# Run the training job
os.system('python fine_tune.py --config=llama.yaml')

# Load model
def load_peft_model_from_checkpoint(model_name: str, model_path: str, task_type: str):
    if task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            return_dict=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif task_type == "SEQ_2_SEQ_LM":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

model_name = "meta-llama/Llama-2-7b-chat"
model_path = "test_llama"
task_type = "CAUSAL_LM"
model, tokenizer = load_peft_model_from_checkpoint(model_name, model_path, task_type)

model_name_for_pipeline = "meta-llama/Llama-2-7b-chat-hf"
tokenizer_for_pipeline = AutoTokenizer.from_pretrained(model_name_for_pipeline)
text_generation_pipeline = transformers.pipeline(
    "text-generation",
    model=model_name_for_pipeline,
    torch_dtype=torch.float16,
    device_map="auto",
)
sequences = text_generation_pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer_for_pipeline.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

prompt = "this is a test"
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
inputs = inputs.to('cuda')
output = tokenizer.decode(
    model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=1,
        repetition_penalty=1,
    )[0],
    skip_special_tokens=True
)
print(output)



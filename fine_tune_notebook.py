import os
import pandas as pd
from huggingface_hub import login
import argparse
import yaml
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from peft import (
    PeftConfig,
    TaskType,
    get_peft_model,
    PeftModel
)

def load_peft_model_from_checkpoint(model_name: str, model_path: str, task_type: str):
    """
    Load a PEFT model from a checkpoint.

    Args:
        model_name (str): The name of the model.
        model_path (str): The path to the model checkpoint.
        task_type (str): The type of task (e.g., "CAUSAL_LM", "SEQ_2_SEQ_LM").

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer for the model.
    """

    if task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            return_dict=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif task_type == "SEQ_2_SEQ_LM":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


def format_message(user_prompt, system_prompt='', history=[]):
    """
    Format a message with a user prompt, system prompt, and conversation history.

    Args:
        user_prompt (str): The user's message.
        system_prompt (str, optional): The system's message. Defaults to ''.
        history (list, optional): The conversation history. Defaults to [].

    Returns:
        str: The formatted message.
    """
    
    message = f"<s>[INST] <<SYS>>\n{{ {system_prompt} }}\n<</SYS>>\n\n"
    for user_msg, model_answer in history:
        message += f"{{ {user_msg} }} [/INST] {{ {model_answer} }} </s><s>[INST] "
    message += f"{{ {user_prompt} }} [/INST]\n"

    return message


def generate_response(model, tokenizer, prompt: str, device='cuda', max_new_tokens=100, temperature=1, repetition_penalty=1):
    """
    Generate a response using the model.

    Args:
        model: The trained model.
        tokenizer: The tokenizer associated with the model.
        prompt (str): The user input prompt.
        device (str): The device to run the model on, default is 'cuda' (GPU). Use 'cpu' if you don't have a GPU.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generating text.
        repetition_penalty (float): Penalty for repeated tokens.

    Returns:
        str: The generated response.
    """
    
    inputs = tokenizer(format_message(prompt), return_tensors="pt", return_token_type_ids=False)
    inputs = inputs.to(device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )[0]
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Load PEFT model and generate a response.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt for generating a response.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["model_name"]
    model_path = config["train"]["final_save_path"]
    task_type = config["task_type"]
    
    model, tokenizer = load_peft_model_from_checkpoint(model_name, model_path, task_type)
    
    response = generate_response(model, tokenizer, args.prompt)
    print(response)


if __name__ == "__main__":
    main()
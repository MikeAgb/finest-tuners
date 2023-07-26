import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import torch
from datasets import Dataset, load_dataset
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator
)

from util import *

logging.basicConfig(level=logging.DEBUG)
os.environ["WANDB_DISABLED"]="true"


def get_model(
		model_name: str,
	    model_config_args: Optional[dict] = None,
	    quant_config_args: Optional[dict] = None) -> AutoModelForCausalLM:
	"""loads hugging face model and returns it
	
	Args:
		model_name: name of model or path to checkpoint, for now restricted to causalLLM
		model_config_args: config dictionary, such as output_attention, etc. 
		quant_config_args: the quantization configuration for BitsAndBytes

	"""

	if model_config_args is None:
		model_config_args={}
	if quant_config_args is None: 
		quant_config_args={}


	model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, use_cache=True, **model_config_args)
	quant_config = BitsAndBytesConfig(**quant_config_args)

	#TODO need to add other types of LLM like seq2seq
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		quantization_config=quant_config,
		trust_remote_code=True,
		device_map="auto",
		offload_folder="offload",
		config=model_config)

	return model


def load_peft_model(
	model,
	peft_config_args: dict,
	peft_type: str = "lora"):
	
	"""gets peft model from base

	Args:
		model: the base model
		peft_config_args: config dicitonary for peft
		peft_type: lora, ia3, prefix

	"""
	assert peft_type in ("lora", "ia3", "prefix"), "peft_type must be a string in (lora, ia3, prefix)"

	#TODO: add other task types
	if peft_type == "lora":
		peft_config = LoraConfig(**peft_config_args)
	elif peft_type == "ia3":
		peft_config = IA3Config(**peft_config_args)
	elif peft_type == "prefix":
		peft_config = PrefixTuningConfig(**peft_config_args)
	
	peft_model = get_peft_model(model, peft_config)
	return peft_model


def tokenize_function(
	example,
    tokenizer,
	truncation,
	max_length,
	padding,
	task_type
	):

	assert task_type in ("CAUSAL_LM"), "only causal_lm supported for now"
	
	if task_type == "CAUSAL_LM":
		example = tokenizer(example['full_prompt'], truncation=truncation, max_length=max_length, padding=padding)

	# for now not supported yet
	elif task_type == "SEQ_2_SEQ_LM":
			example["input_ids"] = tokenizer(example["input_ids"], truncation=truncation, max_length=max_length, padding=padding).input_ids
			example["labels"] = tokenizer(example["labels"], truncation=truncation, max_length=max_length, padding=padding).input_ids

	return example

def tokenize_dataset(
	dataset: Dataset,
	tokenizer,
	max_length: Optional[int] = None,
	truncation: Optional[bool] = True,
	padding: Optional[bool] = True,
	task_type: Optional[str] = "CAUSAL_LM"
	):
	"""apply tokenizer to dataset

	Args:
		dataset: the HuggingFace Dataset to be tokenized
		tokenizer: the HuggingFace Tokenizer
		max_length: the max length of sequence
		truncation: whether to truncate at max length
		padding: apply padding
	"""

	tokenized_dataset =  dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer, 
																	'truncation': truncation, 
																	'padding': padding, 
																	'max_length': max_length,
																	'task_type': task_type}, batched=True)
	return tokenized_dataset


def train_regular(
	model,
	tokenizer,
	trainer_config: dict,
	final_save_path: str,
	train_dataset: Dataset,
	eval_dataset: Optional[Dataset]= None,
	mlm: Optional[bool] = False) -> None:
	"""get training arguments from config dict

	Args
		model: model to be trained
		trainer_config: config dict of training arguments
		final_save_path: where to save the final model
		train_dataset: tokenized training dataset
		eval_dataset: optional evaluation set
	"""

	tokenizer.pad_token = tokenizer.eos_token
	training_args = TrainingArguments(output_dir=final_save_path,**trainer_config)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm),
		)

	trainer.train()

	trainer.model.save_pretrained(final_save_path)
	# tokenizer.save_pretrained(peft_model_path)


#TODO add accelerate library capacity
def train_accelerate():
	pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-path", type=str)
	args, _ = parser.parse_known_args()

	with open(args.config_path, 'r') as f:
		config = yaml.safe_load(f)

	logging.debug(f"config: {config}")

	# turn from string into the torch datatype
	config['model']['quant_config']['bnb_4bit_compute_dtype'] = eval(config['model']['quant_config']['bnb_4bit_compute_dtype'])

	model = get_model(config["model"]["model_name"], config["model"]["model_config"], config["model"]["quant_config"])

	peft = config["peft"]
	if peft:
		model = load_peft_model(model, config["peft"]["peft_config"], config["peft"]["type"])
		task_type = config["peft"]["peft_config"]["task_type"]
	else:
		task_type = config["task_type"]

	trainable_params, all_params = get_trainable_parameters(model)
	logging.debug(f"trainable parameters: {trainable_params / all_params * 100}%")

	tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
	
	if config["data_location"] == "local":
		train_data = pd.read_csv(config['train_data'])
		train_dataset = Dataset.from_pandas(train_data)
		if config['eval_data']:
			eval_data = pd.read_csv(config['eval_data'])
			eval_dataset = Dataset.from_pandas(eval_data)

	elif config["data_location"] == "hub":
		dataset = load_dataset(config["location"])
		train_dataset = dataset["train"]
		eval_dataset = dataset["validation"]


	logging.debug(train_data.head())

	
	train_dataset_tokenized = tokenize_dataset(
		train_dataset,
		tokenizer,
		config['token']['max_length'],
		config['token']['truncation'],
		config['token']['padding'],
		task_type)

	if config['eval_data']:
			eval_dataset_tokenized = tokenize_dataset(
				eval_dataset,
				tokenizer,
				config['token']['max_length'],
				config['token']['truncation'],
				config['token']['padding'],
				task_type)
	else:
		eval_dataset_tokenized=None


	if config["train"]["type"]=="trainer":
		train_regular(model,
					  tokenizer,
					  config["train"]["train_config"],
					  config["train"]["final_save_path"],
					  train_dataset_tokenized,
					  eval_dataset_tokenized,
					  config['train']['mlm'])
	else:
		 raise Exception("only trainer is currently supported")


if __name__ == "__main__":
	main()





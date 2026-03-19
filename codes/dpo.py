from unsloth import PatchDPOTrainer
import pandas as pd
from unsloth import FastLanguageModel
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported
import copy 

PatchDPOTrainer()

#variables
modelname = "/model/llama_3.2_3b_sft"                               # sft model path
dpo_data_path = "/data/llama3b_2e_dpo_data.csv"                           # data path for dpo training
trained_model_path = "/model/llama_3b2e_dpo"                      # new model path after dpo training

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = modelname, # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length = max_seq_length,
    # dtype = dtype,
    # load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

refmodel = copy.deepcopy(model)

tokenizer.pad_token = tokenizer.eos_token

def create_dict_from_dataframe(df):
    # Extract values from the specified columns
    prompt_values = df['instruction'].tolist()
    chosen_values = df['r1'].tolist()
    rejected_values = df['r3'].tolist()

    # Create the dictionary
    result_dict = {'prompt': prompt_values, 'chosen': chosen_values, 'rejected': rejected_values}

    return result_dict

df = pd.read_csv(dpo_data_path)

train_data,valid_data = train_test_split(df,random_state=123,test_size=0.05)

dpo_train_dataset_dict=create_dict_from_dataframe(train_data)
dpo_valid_dataset_dict=create_dict_from_dataframe(valid_data)

train_dataset = Dataset.from_dict(dpo_train_dataset_dict)
valid_dataset = Dataset.from_dict(dpo_valid_dataset_dict)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = refmodel,
    args = DPOConfig(
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 4,
        remove_unused_columns=False,
        # warmup_ratio = 0.1,
        # num_train_epochs = 50,
        learning_rate = 5e-7,
        evaluation_strategy="steps",
        max_steps=1000,
        logging_first_step=True,
        logging_steps = 10,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        # optim = "adamw_8bit",
        # weight_decay = 0.0,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "outputs_llama3b2e",
        # report_to = "none", # Use this for WandB etc
    ),
    beta = 0.6,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    tokenizer = tokenizer,
    precompute_ref_log_probs=False,
    max_length = 1600,
    max_prompt_length = 1000,
    max_target_length=600
)

dpo_trainer.train()

model.save_pretrained(trained_model_path)
tokenizer.save_pretrained(trained_model_path)
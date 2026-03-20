from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load different pretrained models or model path to directly download from huggingface
tokenizers = AutoTokenizer.from_pretrained("/inference/llama_3.2_3b_lora_model")
models = AutoModelForCausalLM.from_pretrained("/inference/llama_3.2_3b_lora_model",torch_dtype=torch.float16,device_map="auto")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

import pandas as pd
df = pd.read_csv("instruct_train.csv")

from tqdm import trange
out = []
for i in trange(df.shape[0]):
    inputs = tokenizers(
    [
        alpaca_prompt.format(
            df.loc[i]["instruction"], # instruction
            df.loc[i]["input"], # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    
    # outputs = models.generate(**inputs, max_new_tokens = 200, use_cache = True,temperature=0.7,top_k=50,top_p=0.9)
    outputs = models.generate(
    inputs["input_ids"],
    max_new_tokens=100,  # Maximum length of output sequence
    use_cache = True,
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # Prevent repeating n-grams
    do_sample=False,  # Use sampling instead of greedy decoding
    top_k=10,  # Keep only top k tokens with highest probability
    top_p=0.7,  # Keep the top tokens with cumulative probability >= top_p
    temperature=0.5,  # Higher temperature means more random outputs
    )
    # outputs = tokenizers.batch_decode(outputs)
    outputs = tokenizers.decode(outputs[0], skip_special_tokens=True)
    # print("Output",df.loc[i]["output"])
    a = outputs.find("Response")
    outputs = outputs[a+10:]
    # print("Generated",outputs)
    out.append(outputs)

df["Generated"] = out
df.to_csv("llama3b_t0.5_k10_p0.7auto.csv",index=False)
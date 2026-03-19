from transformers import pipeline
import spacy
from evaluate import load
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from tqdm import tqdm
import pandas as pd

# Reward 1: Severity Score
severity_model_path = "/model/severity_mb"     # trained modern bert model for reward score calculation
model_severity = AutoModelForSequenceClassification.from_pretrained(severity_model_path).to("cuda")
tokenizer_severity = AutoTokenizer.from_pretrained(severity_model_path)

def severity_class(input,output):
    reward_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 16
    }
    query_response_pairs = []
    for input, output in zip(input, output):
        a = f"""<Medical_Report>
        {input}
        </Medical_Report>

        <Impression>
        {output}
        </Impression>

        You are a medical practitioner responsible for assessing the severity of diseases based on the patient's medical condition
        as detailed in the provided report and impressions. Your task is to analyze the report carefully and classify the severity
        level on a scale of 0 to 3, where:
        0 indicates the absence of the disease (Absent),
        1 indicates a mild condition (Mild),
        2 indicates a moderate condition (Moderate),
        3 indicates a severe condition (Severe).
        Ensure that your assessment is based on clinical indicators, lab results, imaging findings, and any relevant medical impressions
        present in the report. Provide only the numerical severity index as the output, without any additional text or explanation.
        """
        query_response_pairs.append(a)
        # if len(query_response_pairs)==5:
        #     break

    class_tensors = []
    for rp in tqdm(query_response_pairs):
        inputs = tokenizer_severity(rp,return_tensors="pt").to("cuda")
        outputs = model_severity(**inputs)
        reward = torch.argmax(outputs.logits, dim=1)
        class_tensors.append(reward.item())

    return class_tensors

def severity_reward(batch):
    gt = severity_class(batch["input"], batch["output"])
    pred = severity_class(batch["input"], batch["Generated"])
    # print(gt)
    # print(pred)
    scores = []
    for g,p in zip(gt,pred):
        s = (g-p)**2
        scores.append(s)
    return scores
    
# Reward 2: FBR Score
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the model and tokenizer directly
model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).cuda()

# Function to perform token classification
def biomedical_ner(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)#.cuda()
    inputs = inputs.to('cuda')

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)

    # Process results
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_predictions = [model.config.id2label[prediction.item()] for prediction in predictions[0]]

    # Combine tokens and predictions, filtering out special tokens
    results = []
    for token, prediction in zip(tokens, token_predictions):
        # Skip special tokens like [CLS], [SEP], etc.
        if token.startswith('[') and token.endswith(']'):
            continue

        # Handle wordpiece tokens (tokens starting with ##)
        if token.startswith('##') and prediction != "O":
            # Append to the previous token
            if results:
                results[-1] += token[2:]
                # results[-1]["word"] += token[2:]
            continue

        # Add regular tokens with their predictions
        if prediction != "O":  # Only include entities, not "Outside" tokens
            results.append(token)
            # results.append({
            #     "entity": prediction,
            #     "word": token,
            #     "score": float(torch.max(torch.softmax(outputs.logits[0][len(results)], dim=0)).item())
            # })

    return results

def ner(batch):
    scores_fbr = []
    scores_c = []
    scores_t = []
    scores_h = []
    for output, generated in tqdm(zip(batch["output"], batch["Generated"])):
        
        l_gen = len(generated.split())
        
        output = biomedical_ner(output)
        generated = biomedical_ner(generated)

        l_ner_gt = len(output)
        l_ner_pred = len(generated)
        output = set(output)
    
        intersect_words = 0
        for g in generated:
            if g in output:
                intersect_words += 1
    
        scores_fbr.append(intersect_words/l_ner_gt)
        scores_c.append(l_ner_pred/l_ner_gt)
        scores_t.append(l_ner_pred/l_gen)
        scores_h.append(intersect_words/l_gen)
        # break
    return scores_fbr, scores_c, scores_t, scores_h


# Reward 3: BERTScore
def res_fluency(batch):
    contextual_consistency_scores = []
    bertscore = load("bertscore")
    for q, r in tqdm(zip(batch["input"], batch["Generated"])):
        results = bertscore.compute(predictions=[q], references=[r], lang="en")
        result = results['f1'][0]
        contextual_consistency_scores.append(result)

    reward_tensors = [reward for reward in contextual_consistency_scores]
    return reward_tensors


# Reward 4: Diversity Score
nlp = spacy.load("en_core_web_sm")

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def diversity(batch):
    jacc_dis = []
    for q, r in tqdm(zip(batch["output"], batch["Generated"])):
        str1 = normalize(q, nlp)
        str1 = set(str1.split())
        str2 = normalize(r, nlp)
        str2 = set(str2.split())
        sim_score = (float(len(str1 & str2)) / len(str1 | str2))
        jacc_dis.append(sim_score)

    reward_tensors = [reward for reward in jacc_dis]

    return reward_tensors


# Score Generation
# old path, new path
datapaths = [
    ["/rado_new_rewards/llama3b_2e_0.6t_auto.csv","llama3b_2e_0.6t_auto_rewards.csv"],
    ["/rado_new_rewards/llama3b_2e_1.2t_auto.csv","llama3b_2e_1.2t_auto_rewards.csv"],
    ["/rado_new_rewards/llama3b_2e_1.8t_auto.csv","llama3b_2e_1.8t_auto_rewards.csv"],
    ["/rado_new_rewards/llama3b_2e_unsloth_res.csv","llama3b_2e_unsloth_res_rewards.csv"]
]

for data_path,new_data_path in datapaths:
    df = pd.read_csv(data_path)
    df.dropna(inplace = True)
    df.rename(columns={'generated_response':'Generated'}, inplace=True)
    data = df.to_dict(orient="list")
    
    print("Data Path", data_path)
    fbr,c,t,h = ner(data)
    df["fbr"] = fbr
    df["C"] = c
    df["T"] = t
    df["H"] = h
    print("NER done")
    
    o2 = severity_reward(data)
    df["M"] = o2
    print("Severity done")
    
    o3 = res_fluency(data)
    df["res_fluency"] = o3
    print("Response Fluency Done")
    
    o4 = diversity(data)
    df["diversity"] = o4
    print("Diversity Done")
    
    df.to_csv(new_data_path,index = False)
    print("Data saved at new data path",new_data_path)
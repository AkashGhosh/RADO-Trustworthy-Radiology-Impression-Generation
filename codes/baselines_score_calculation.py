import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load
from transformers import AutoTokenizer, AutoModel
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
#from torchmetrics.text.bert import BERTScore
from torchmetrics.functional.text.bert import bert_score
import random
from sentence_transformers import SentenceTransformer
Sem_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Data set contains both the generated summaries and the gold summaries.
new_data_path = "/data_path"

df_new = pd.read_csv(new_data_path)
df_new.dropna(inplace=True)

def get_scores(reference_list: list,
               hypothesis_list: list):
    #val_df = pd.DataFrame(list(zip(hypothesis_list, reference_list)), columns=['Gold_summary', 'predicted_summary'])
    #file_name = "/mnt/Data/akashghosh/MDS/Data/" + "MDS_TI_encoderdecoder_resnet_.csv"
    #val_df.to_csv(file_name, index=False) 
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    bs = 0
    J = 0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([word_tokenize(reference)], word_tokenize(hypothesis))

        Ref_E = Sem_model.encode(reference)
        Hyp_E = Sem_model.encode(hypothesis)

        bs += cosine_similarity([Ref_E],[Hyp_E])

        reference = reference.split()
        hypothesis = hypothesis.split()


        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        "meteor": met*100/count,
        "BS": bs/count
    }

reference_list=df_new['Generated'].tolist()
golden_list=df_new['output'].tolist()

scores = get_scores(reference_list,golden_list)

print(scores)
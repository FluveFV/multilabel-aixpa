import os

import torch
import numpy as np
import argparse
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import langid

if not torch.cuda.is_available():
    raise Exception('CUDA apparently not available.')

parser = argparse.ArgumentParser(description="Predict which labels for input text.")

parser.add_argument(
    "-k",
    type=int,
    default=3,
    help="Number of action categories to predict"
)

parser.add_argument(
    "--map",
    action="store_true",
    help="Enable mapping from category of the action to relative macrocategory and field"
)

args = parser.parse_args()
k = args.k
map = args.map

#By default, the trained model is stored in a multi_label_classification or
# single_label_classification, depending on how your data appears

#In case the tokenizer and the model are two different deep learning architectures,
#they are stored in different files.
t_file_path = "multi_label_classification/tokenizer_config"
#t_file_path = "single_label_classification/tokenizer_config"
tokenizer = AutoTokenizer.from_pretrained(t_file_path)


m_file_path = ("multi_label_classification/model_config")
#m_file_path = ("single_label_classification/model_config")
config = AutoConfig.from_pretrained(m_file_path)
model = AutoModelForSequenceClassification.from_pretrained(m_file_path, config=config)
model.eval()

def sigmoid(x):  #necessary for multilabel
    return 1 / (1 + np.exp(-x))

threshold = float(input("Enter problem activation threshold (close to 0.1 for organization data, close to 0.7 for municipalities data):"))

new_example = '''
sito internet . realizzare il sito internet ed attivar, dal punto di vista tecnico, 
i link a siti di interesse per la conciliazione e creazione di un'area tematica 
(o una ""pagina"") dedicata ad aspetti normativi, family audit ecc. per la conciliazione . 
incrementare conoscenza aspetti conciliazione ed agevolarne eventuale approfondimento sia 
per visitatori sito che per dipendenti
'''

inputs = tokenizer([new_example], return_tensors="pt", truncation=True, padding=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits

probabilities = sigmoid(logits.numpy())

predicted_labels = (probabilities > threshold).astype(int).tolist()

print(probabilities, predicted_labels)
print()
print("Converting to original labels:")
print()
positions = [i for i, val in enumerate(predicted_labels[0]) if val == 1]    #positions of predicted classes
print(positions)
label2id = config.label2id
output = [label2id[str(i)] for i in positions]
print(label2id)
print(output)


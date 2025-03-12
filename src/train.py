## based on code from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
import time
import datasets
from datasets import load_dataset, ClassLabel, DatasetDict
import torch
import os
from os import path, makedirs
import numpy as np
from numpyencoder import NumpyEncoder
import pandas as pd
import json

import wandb

#torch.cuda.empty_cache()
#os.environ["WANDB_MODE"]='offline'
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

project_name = "Mar25"
wandb.init(project=f"{project_name}", name=f"bbi_poly_scheduler_aziende", config={})

from transformers import AutoTokenizer
#📂
class Dataloader:
    def __init__(self, file_path, test_size=0.2, val_size=0.18, random_state=25,
                 model_path='microsoft/deberta-large', **kwargs):
        '''
        Args:
            file_path (str): Path to the file.
            test_size (float): Proportion of the dataset to use as the test set.
            val_size (float): Proportion of the train/validation split to use as the validation set.
            random_state (int): Seed for reproducibility.
        '''

        self.file_path = file_path
        self.file_type = file_path.split('.')[-1]  # Data needs to be stored locally as a file
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       use_fast=True,
                                                       use_cache=False)

        self.dataset = None
        self.class_weights = None
        self.label_columns = None
        self.problem_type = None
        self.label_column_names = None

        self.run()
        # printing example of data
        print(self.dataset['train'][0]) #D
        #print(torch.tensor(self.dataset["train"][-2]["labels"]).shape)

    def load_data(self):
        loaders = ["json", "csv", "gzip"]
        if self.file_type not in loaders:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        self.file_type = 'parquet' if self.file_type == 'gzip' else self.file_type.split('.')[-1]
        return load_dataset(path=self.file_type, data_files=f'{self.file_path}')

    def map_labels(self):
        '''
        Creates maps to connect original label distribution to
        a dense distribution and viceversa.

        In the original formulation, the data has 8-digit labels for a taxonomy of hundreds of labels
        labels in total, raising issues.
        '''
        self.unique_classes = set()
        for col in self.dataset["train"].features:
            if col.lower().startswith("label"):
                self.unique_classes.update(pd.Series(self.dataset["train"][col]).dropna().unique())
        self.num_classes = len(self.unique_classes)
        self.id2class = {int(label): idx for idx, label in enumerate(self.unique_classes)}
        self.class2id = {idx: int(label) for idx, label in enumerate(self.unique_classes)}
        self.save_encoding()

    def save_encoding(self):
        with open("reverse_encoding.json", "w") as f:
            json.dump(self.class2id, f,
                      indent=4, sort_keys=True,
                      separators=(', ', ': '), ensure_ascii=False,
                      cls=NumpyEncoder)

    def convert_labels(self, example):
        '''
        Converts the original distribution of labels to a dense distribution
        using the previously defined mapping.
        Iterates on the elements inside the all_labels column
        '''
        for col in self.label_names:
            if col.lower().startswith("label") and isinstance(example[col], (float, int)):
                key = example[col]
                example[col] = self.id2class[key]

        return example

    def merge_labels(self, example):
        '''
        Takes one row and combines
        multiple label columns into a single `all_labels` list.
        '''
        label_names = [col for col in self.dataset["train"].features if col.lower().startswith("label")]
        all_labels = [int(example[col]) for col in label_names if isinstance(example[col], (int, float))]
        example["all_labels"] = all_labels  # still in raw form, original distribution of IDs
        return example

    def stratified_split(self):
        '''
        Splits the dataset using the one column's composition to make proportionate training and testing sets.
        This is required in taxonomies with many classes of varying frequencies.
        '''

        first_column = [col for col in self.label_names if col.lower().startswith("label")][0]

        self.dataset = self.dataset.class_encode_column(first_column)

        train_test_split = self.dataset["train"].train_test_split(
            test_size=self.test_size,
            stratify_by_column=first_column,
            shuffle=True,
            seed=self.random_state,
        )
        train_set = train_test_split["train"]
        test_set = train_test_split["test"]

        self.dataset = DatasetDict({'train': train_set,
                                    'test': test_set})

    def tokenizing(self, example):
        '''
        Tokenize text, while labels get one-hot encoded
        '''
        text = example["text"]
        example = self.tokenizer(text, truncation=True, max_length=512)
        return example

    def hot_encoding(self, example):
        all_labels = example['all_labels']
        labels = [0. for i in range(self.num_classes)]
        for i in all_labels:
            labels[int(i)] = 1.
            # On an N-length vector where N is the number of classes,
            # the position correspondent to class "i" becomes 1, the rest is 0
        example["labels"] = labels # vector added to tokenized element
        return example

    def run(self):
        self.dataset = self.load_data()

        self.label_names = self.dataset['train'].features

        self.map_labels()  # creates maps (does NOT apply them)

        self.dataset = self.dataset.map(self.convert_labels)

        self.dataset = self.dataset.map(self.merge_labels)  # merges labels

        # Mapping labels into dense distribution +
        # splitting into train and test
        self.stratified_split()

        # D
        #If there is at least one observation with multiple labels, then
        # it will be one hot encoded for Binary cross entropy loss, otherwise
        # plain cross entropy for multiclass problems
        multi_label_instances = list(filter(lambda x: len(x['all_labels']) > 1, self.dataset['train']))

        if len(multi_label_instances) > 1:
            self.problem_type = 'multi_label_classification'
            self.dataset['train'] = self.dataset['train'].map(self.hot_encoding)
            self.dataset['test'] = self.dataset['test'].map(self.hot_encoding)
            self.label_column_names = 'labels'

        else:
            self.problem_type = 'single_label_classification'
            self.label_column_names = 'label'
            #self.dataset['train'] = self.dataset['train'].rename_column('all_labels', 'labels')

        self.dataset['train'] = self.dataset['train'].map(self.tokenizing)

        self.dataset['test'] = self.dataset['test'].map(self.tokenizing)

        print(f'Classification problem identified as {self.problem_type}')
        print()

####### Change model here ########################
model_path = 'dbmdz/bert-base-italian-xxl-uncased'
##################################################

###### Change input data here ####################
datasets.disable_progress_bars()
file_path = 'data/addestramento_aziende.csv'
dl = Dataloader(file_path=file_path, model_path=model_path)
print('Dataset in use:', file_path)
##################################################

tokenized_dataset = dl.dataset
num_classes = dl.num_classes
id2class = dl.id2class
class2id = dl.class2id
tokenizer = dl.tokenizer
problem_type = dl.problem_type
label_names = dl.label_column_names

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

def dataframe_append(directory, data_filename, row):
    # module for use in different stages of training and testing
    if not path.exists(model_dir):
        makedirs(model_dir)
    FN = directory + '/' + data_filename

    if not (path.exists(FN)):
        row.to_csv(FN, index=False, header=True)
    else:
        row.to_csv(FN, mode='a', index=False, header=False)

def sigmoid(x):
   return 1/(1 + np.exp(-x))

from torch import nn
import torch.nn.functional as F

def compute_metrics(eval_pred, model_dir="./model"):
    result = {}
    metrics = ['precision', 'recall', 'f1']

    if problem_type == 'multi_label_classification':  #problem_type is a global var
        logits, labels = eval_pred
        probabilities = sigmoid(logits)
        #for this loss, documentation indicates normalized logits (probabilities)
        bce_loss = F.binary_cross_entropy(torch.tensor(probabilities), torch.tensor(labels), reduction="mean").item()
        result['eval_loss'] = bce_loss

        #for performance metrics we need the labels predicted
        predictions = (probabilities > 0.3).astype(int)
        for i in metrics:
            metric_function = evaluate.load(i, "multilabel")
            result[i] = metric_function.compute(
                predictions=predictions,
                references=labels,
                average='weighted'
            )[i]

    elif problem_type == 'single_label_classification':
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        softmax = nn.Softmax(dim=-1)
        probabilities = softmax(logits)
        predictions = torch.argmax(probabilities, dim=-1)
        #for this loss, documentation indicates unnormalized output (logits)
        ce_loss = F.cross_entropy(logits, labels, reduction="mean").item()
        result['eval_loss'] = ce_loss

        for i in metrics:
            metric_function = evaluate.load(i, "multiclass")
            result[i] = metric_function.compute(
                predictions=predictions,
                references=labels,
                average='weighted'
            )[i]

    #print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")  # D

    row = pd.DataFrame({k: [round(v, 5)] for k, v in result.items()})
    if not path.exists(model_dir):
        makedirs(model_dir)
    data_filename = 'train_metrics.csv'
    dataframe_append(directory=model_dir, data_filename=data_filename, row=row)

    return result

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=num_classes,
    id2label=id2class, label2id=class2id,
    problem_type=problem_type)

model_dir = problem_type

if not path.exists(model_dir):
    makedirs(model_dir)
else:
    print('Overwriting existing checkpoints')

training_args = TrainingArguments(
    run_name = model_dir + time.strftime("%y_%m_%d-%H_%M_%S", time.localtime()),
    output_dir=model_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=50,
    max_steps=-1, #for debugging, remove minus sign
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    weight_decay=1e-3,
    save_total_limit=1,
    metric_for_best_model = "eval_loss",
    load_best_model_at_end=True,
    fp16=False,
    warmup_ratio=0.2,
    learning_rate=1.5e-4,
    lr_scheduler_type="linear",
    #report_to=None,  #Wandb and other are excluded with this setting
    # Parallel computing parameters (GPU)
    #dataloader_num_workers = 4,
    #ddp_find_unused_parameters=False,
    #ddp_backend='nccl',
    resume_from_checkpoint = False
)

if problem_type=='multi_label_classification':
    training_args.label_names = [label_names]  #by adding a specific column name it avoids a known bug in transformers

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
)

print("Starting training...")
trainer.train()
print("Training finished")
trainer.evaluate()
final_path = f'{model_dir}/'
print()
print(f"Model is being saved to dir: {final_path}")
print()
if not path.exists(final_path):
    makedirs(final_path)
else:
    print('Overwriting previously saved model')

trainer._load_best_model()

trainer.save_model(final_path + 'model_config')
tokenizer.save_pretrained(final_path + 'tokenizer_config')

test_prediction = trainer.predict(tokenized_dataset['test'])
logits = test_prediction.predictions
true_labels = test_prediction.label_ids
probabilities = sigmoid(logits)

predicted_labels = (probabilities > 0.3).astype(int)
print(f'Test predictions are saved in {model_dir}/predictions.csv')
pd.DataFrame({
    "true_labels": true_labels.tolist(),
    "predicted_labels": predicted_labels.tolist(),
    "probabilities": probabilities.tolist()
}).to_csv(model_dir + '/predictions.csv', index=False)


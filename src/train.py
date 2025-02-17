## based on code from: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

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

project_name = "Feb25"
wandb.init(project=f"{project_name}", name=f"multilabel_raw", config={})

from transformers import AutoTokenizer
#📂
class Dataloader:
    def __init__(self, file_path, test_size=0.2, val_size=0.25, random_state=25,
                 model_path='microsoft/deberta-v3-small', **kwargs):
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

        self.run()
        # printing example of data
        print(self.dataset['train'][-2])

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
        all_labels = example['all_labels']

        labels = [0. for i in range(self.num_classes)]
        for i in all_labels:
            labels[int(i)] = 1.
            # On an N-length vector where N is the number of classes,
            # the position corrispondent to class i becomes 1 if in the data it is so.

        example = self.tokenizer(text)
        example["labels"] = labels  # vector added to tokenized element
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

        # tokenization
        self.dataset['train'] = self.dataset['train'].map(self.tokenizing)
        self.dataset['test'] = self.dataset['test'].map(self.tokenizing)

model_path = 'microsoft/deberta-v3-small'

dl = Dataloader(file_path='data/data_d.gzip', model_path=model_path)
tokenized_dataset = dl.dataset
num_classes = dl.num_classes
id2class = dl.id2class
class2id = dl.class2id
tokenizer = dl.tokenizer

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

def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["f1", "accuracy", "precision", "recall"])
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    if not path.exists(model_dir):
        makedirs(model_dir)
    data_filename = 'train_metrics.csv'
    result = clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
    row = pd.DataFrame({k: [v] for k, v in result.items()})

    dataframe_append(directory=model_dir, data_filename=data_filename, row=row)

    return result

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=num_classes,
    id2label=id2class, label2id=class2id,
    problem_type = "multi_label_classification")

model.gradient_checkpointing_enable()

model_dir = 'DeBERTa'

if not path.exists(model_dir):
    makedirs(model_dir)
else:
    print('Overwriting existing checkpoints')

training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=25,
    #max_steps=1, #D
    weight_decay=0.005,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    #report_to=None,  #Wandb and other are excluded with this setting
    fp16=True,
    optim="adamw_torch_fused",
    # parallel computing parameters (GPU)
    #dataloader_num_workers = 4,
    #ddp_find_unused_parameters=False,
    #ddp_backend='nccl',
    resume_from_checkpoint = False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Starting training...")
trainer.train()
print("Training finished")

final_path = f'{model_dir}/multiclassifier_config'
print()
print(f"Model is being saved to dir: {final_path}")
print()
if not path.exists(final_path):
    makedirs(final_path)
else:
    print('Overwriting previously saved model')

trainer.save_model(final_path + 'model')
tokenizer.save_pretrained(final_path + 'tokenizer')

test_prediction = trainer.predict(tokenized_dataset['test'])
logits = test_prediction.predictions
true_labels = test_prediction.label_ids
probabilities = sigmoid(logits)

predicted_labels = (probabilities > 0.3).astype(int)
print(f'Test predictions are saved in {model_dir}/predictions.csv')
pd.DataFrame({"true_labels": true_labels.tolist(),
              "predicted_labels": predicted_labels.tolist()
              }).to_csv(model_dir + '/predictions.csv', index=False)
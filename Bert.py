from google.colab import drive
drive.mount('/content/drive')

!pip install torch -U
!pip install transformers
%cd /content/drive/MyDrive         

!git clone https://github.com/guillaumejaume/FUNSD.git

def process_document(data):
    doc_text = []
    doc_labels = []

    for block in data['form']:
        for item in block['words']:
            doc_text.append(item['text'])
            doc_labels.append(block['label'])

    return doc_text, doc_labels
    
import json
import os
import zipfile

# Extract the dataset
with zipfile.ZipFile('/content/drive/MyDrive/FUNSD/dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/FUNSD')

def load_funsd_data(data_dir):
    texts = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                doc_text, doc_labels = process_document(data)
                texts.append(doc_text)
                labels.append(doc_labels)

    return texts, labels

# Replace the following paths with the correct paths to the training and testing data directories in the extracted dataset
train_data = '/content/drive/MyDrive/FUNSD/dataset/training_data/annotations'
test_data = '/content/drive/MyDrive/FUNSD/dataset/testing_data/annotations'
#train_labels_dir = '/content/drive/MyDrive/FUNSD/dataset/training_data/images'
#test_labels_dir = "/content/drive/MyDrive/FUNSD/dataset/testing_data/images"

train_texts, train_labels = load_funsd_data(train_data)
test_texts, test_labels = load_funsd_data(test_data)

### Bert Code
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from albumentations import Compose, Rotate, RandomCrop, Flip, Transpose
import json
import os
import zipfile

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Extract the dataset
with zipfile.ZipFile('/content/drive/MyDrive/FUNSD/dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/FUNSD')

def process_document(document):
    text = document.get('text', '') # Use the `get` method to return an empty string if the 'text' key doesn't exist
    annotations = document.get('annotations', []) # Use the `get` method to return an empty list if the 'annotations' key doesn't exist
    labels = []

    for annotation in annotations:
        if annotation['label'] != "Text":
            continue
        start_idx = annotation['start_offset']
        end_idx = annotation['end_offset']
        label = annotation['value']
        labels.extend(['O'] * start_idx)
        if len(label.split()) == 1:
            labels.extend(['B-' + label])
        else:
            labels.extend(['B-' + label.split()[0]])
            labels.extend(['I-' + label.split()[1]] * (len(label.split()) - 1))
        labels.extend(['O'] * (len(text) - end_idx))

    return text, labels

def load_funsd_data(data_dir):
    texts = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                doc_text, doc_labels = process_document(data)
                texts.append(doc_text)
                labels.append(doc_labels)

    return texts, labels

# Define the training and testing data directories
train_data_dir = '/content/FUNSD/dataset/training_data/annotations'
test_data_dir = '/content/FUNSD/dataset/testing_data/annotations'

# Load the training and testing data
train_texts, train_labels = load_funsd_data(train_data_dir)
test_texts, test_labels = load_funsd_data(test_data_dir)

import itertools

# Flatten the list of lists of labels
train_labels = list(itertools.chain.from_iterable(train_labels))
test_labels = list(itertools.chain.from_iterable(test_labels))

# Define the label map
label_list = sorted(set(train_labels))
label_map = {label: i for i, label in enumerate(label_list)}

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
class FunsdDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize the input text and labels
        encoding = self.tokenizer.encode_plus(
            text,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert labels to IDs based on the label_map
        label_ids = [self.label_map[label] for label in labels]
        label_ids = label_ids[:self.max_length] + [self.label_map["PAD"]] * (self.max_length - len(label_ids))
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label_ids,
        }



# Fine-tuning settings
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model.to(device)

optimizer = AdamW(roberta_model.parameters(), lr=3e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    roberta_model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device) 
        labels = batch["labels"].to(device) 

        outputs = roberta_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluation loop
    roberta_model.eval()
    preds = []
    true_labels = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = roberta_model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits.detach().cpu().numpy()
        label_ids = labels.detach().cpu().numpy()
        preds.extend(list(np.argmax(logits, axis=2).flatten()))
        true_labels.extend(list(label_ids.flatten()))

    # Print the classification report
    id2label = {v: k for k, v in label_map.items()}
    pred_labels = [id2label[id] for id in preds]
    true_labels = [id2label[id] for id in true_labels]

    print(classification_report(true_labels, pred_labels, digits=4))

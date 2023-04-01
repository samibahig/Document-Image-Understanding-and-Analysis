%%capture
!wget https://guillaumejaume.github.io/FUNSD/dataset.zip -O dataset.zip
!unzip dataset.zip

import json
from pathlib import Path
import cv2
import numpy as np     

def candidates(path):
  anot=json.loads(open(path).read())
  img=Path(path).stem+'.png'
  path_image=Path(path).parent.parent/'images'/img
  h,w,_=cv2.imread(str(path_image)).shape
  question, answer=[],[]
  text={}
  for block in anot['form']:
    if block['label']=='question':
      question.append([block['id'],block['box']])
      text[block['id']]=block['text']
    if block['label']=='answer':
      answer.append([block['id'],block['box']])
      text[block['id']]=block['text']
  # We have built 2 list with answers and questions' info 
  #For each answer we look for its question
  dic={}
  for a in answer:
    bbox=a[1]
    candidates=[]
    candidates_more=[]
    x_a=int(bbox[0])
    x1_a=int(bbox[2])
    y_a=int(bbox[1])
    y1_a=int(bbox[3])
    pto_a=[(x_a+x1_a)/2,(y_a+y1_a)/2]
    for q in question:
      bbox=q[1]
      x_q=int(bbox[0])
      x1_q=int(bbox[2])
      y_q=int(bbox[1])
      y1_q=int(bbox[3])
      pto_q=[(x_q+x1_q)/2,(y_q+y1_q)/2]
      if x_q<x1_a+0.05*w and y1_q>y_a-0.1*h and y_q<y1_a+0.01*h:
        dist=np.sqrt((x_a-x1_q)**2+(pto_a[1]-pto_q[1])**2)
        candidates.append([q[0],dist])
      if x_q<x1_a +0.05*w and y1_q>y_a-0.6*h and y_q<y1_a+0.03*h:
        dist=np.sqrt((x_a-x1_q)**2+(pto_a[1]-pto_q[1])**2)
        candidates_more.append([q[0],dist])
    if candidates!=[]:
      dic[a[0]]=candidates
    else:
      dic[a[0]]=candidates_more
  return dic,text

from operator import itemgetter
def rank(path):
  cand,text=candidates(path)
  for c in cand:
    aux=cand[c]
    cand[c]=[l[0] for l in sorted(aux, key=itemgetter(1))]
  return cand,text

def candidatesANDlabels(path):
  anot=json.loads(open(path).read())
  cand,text=rank(path)
  dic_label={}
  for block in anot['form']:
    if block['label']=='answer':
      id=block['id']
      id_question=[]
      for link in block['linking']:
        if link[0]==id:
          id_question.append(link[1])
        else: id_question.append(link[0])
      lista=cand[id]
      labels=[lista[i] in id_question for i in range(0,len(lista))]
      dic_label[id]=labels
  return cand,dic_label,text     

import os
class Dataset():
    def __init__(self, path_annotation):
        self.path_annotation= path_annotation

    def __iter__(self):
      with os.scandir(self.path_annotation) as files:
        for file in files:
          yield file.name
          
    def __len__(self):
      i=0
      for file in self:
        i+=1
      return i

    def textList(self, o):
      path=self.path_annotation+'/'+o
      cand,label,text=candidatesANDlabels(path)
      question_answer,et=[],[]
      dic={True:1,False:0}
      for c in cand:
        question_answer=question_answer+[text[x]+' '+text[c] for x in cand[c]]
        et=et+[dic[z] for z in label[c]]
      return question_answer,et
    
    def preparation(self):
      text=[]
      label=[]
      for file in self:
        txt,lbl=self.textList(file)
        text=text+txt
        label=label+lbl
      return (text,label)
dataset_train=Dataset('dataset/training_data/annotations')
dataset_test=Dataset('dataset/testing_data/annotations')

text_train, labels_train=dataset_train.preparation()
text_test, labels_test=dataset_test.preparation()

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(text_train, labels_train, test_size=.2)

path='/content/dataset/training_data/annotations/'
i=0
for f in dataset_train:
  f=path+f
  cand,_,_=candidatesANDlabels(f)
  for c in cand:
    if cand[c]==[]:
      i+=1
print(i)

path='/content/dataset/training_data/annotations/'
i,j=0,0
for f in dataset_train:
  f=path+f
  cand,lbl,text=candidatesANDlabels(f)
  for k in lbl:
    j+=1
    if not(True in lbl[k]):
      i+=1
print(i,j)

import torch
class FUNDSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
      
%%capture
pip install transformers

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
 
import numpy as np
from sklearn.metrics import average_precision_score

def mAP_x(scores, labels):
  m=average_precision_score(labels, scores)
  if np.isnan(m):
    return 0
  else:
    return m

def mAP(dataset,coef_FINAL,labels):
  i=0
  map=[]
  for f in dataset:
    cand,txt=rank(dataset.path_annotation+ '/' +f)
    n=len(cand)
    if n>0:
      for c in cand:
        j=i+len(cand[c])
        map.append(mAP_x(coef_FINAL[i:j],labels[i:j]))
        i=j
  return sum(map)/len(map)

import numpy as np
def mRank_x(predictions, labels,coef):
  if predictions==labels:
    return 0
  else:
    if 1 in labels:
      if np.count_nonzero(labels)>1:
        indices = [i for i, x in enumerate(labels) if x == 1]
        s,m=0,0
        for i in indices:
          s=s+sum([c>coef[i] for c in coef])-m
          m+=1
        return s
      else:
        k=labels.index(1)
        return sum([c>coef[k] for c in coef])
    else:
      return len(predictions)

def mRank(dataset,prediction,labels,coef):
  mrank=[]
  i=0
  for f in dataset:
    cand,txt=rank(dataset.path_annotation+ '/' +f)
    n=len(cand)
    if n>0: #comprobamos que el documento tenga algun par clave-valor
      for c in cand:
        j=i+len(cand[c])
        if j!=i:
          mrank.append(mRank_x(prediction[i:j],labels[i:j],coef[i:j]))
          i=j
  return sum(mrank)/len(mrank)

from transformers import BertForSequenceClassification,AutoTokenizer,Trainer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
model = BertForSequenceClassification.from_pretrained('microsoft/layoutlm-base-uncased', num_labels=2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(text_test, truncation=True, padding=True)

train_dataset = FUNDSDataset(train_encodings, train_labels)
val_dataset = FUNDSDataset(val_encodings, val_labels)
test_dataset = FUNDSDataset(test_encodings, labels_test)

args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train() 

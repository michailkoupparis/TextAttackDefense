import string
import nltk
import nlp
import csv
import os
import math 
from textattack.shared.utils import words_from_text
import numpy as np

def get_tokenize_sentences(premises,hypothesis):
  all_sentences = []

  for i in zip(premises,hypothesis):
    st = i[0] + " " + i[1]
    tokens = words_from_text(st)
    all_sentences.append(tokens)
  
  return all_sentences

def return_multi_nli_train_dict(dataset):

  train_data = nlp.load_dataset(dataset)['train']

  train_dict = {'premise':[],'hypothesis':[]}

  perc_ = math.ceil(len(train_data)*0.15)
  print("----EVAL DATASET USING : " + str(perc_) + "\n")

  eval_indices = np.sort(np.random.RandomState(21).choice(np.arange(len(train_data)),size=perc_,))
  train_indices = np.delete(np.arange(len(train_data)),eval_indices)

  train_dict['premise'] = np.array(train_data['premise'])[train_indices] 
  train_dict['hypothesis'] = np.array(train_data['hypothesis'])[train_indices]

  return train_dict

def create_freq_corpus(dataset,csvname):


  if dataset is 'multi_nli':
   print('MULTI') 
   train_data = return_multi_nli_train_dict(dataset)

  else:

    train_data = nlp.load_dataset(dataset)['train']

  
  all_sentences = get_tokenize_sentences(train_data['premise'],train_data['hypothesis'])

  freq_dict = {}

  for sentence in all_sentences:
    for token in sentence:
      if token  in freq_dict:
        freq_dict[token] += 1
      else:
        freq_dict[token] = 1 

  print("WRITE")
  with open(csvname, 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in freq_dict.items():
      writer.writerow([key, value])
  

  return freq_dict

def load_csv(filename):

  with open(filename) as csv_file:
    reader = csv.reader(csv_file)
    freq_dict = dict(reader)
  
  return freq_dict
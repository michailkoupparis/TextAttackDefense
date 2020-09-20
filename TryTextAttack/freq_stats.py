from corpus import get_tokenize_sentences

import pandas as pd
from ast import literal_eval
import numpy as np
import math 

def phi_value(freq):
  return math.log(1+freq)

def get_phi_values(words,freq_dict,in_dict=False):

  if in_dict:
    return [phi_value(int(freq_dict[word])) for word in words if word in freq_dict]

  return [phi_value(int(freq_dict[word])) if word in freq_dict else 0 for word in words]


def get_attack_freq_stats(filename,freq_dict):
  
  attack_data = pd.read_csv(filename)

  premises = [i['premise'] for _, i in attack_data.iterrows()] 
  attacked_hypo = [i['attacked_hypothesis'] for _, i in attack_data.iterrows()] 
  original_hypo = [i['original_hypothesis'] for _, i in attack_data.iterrows()] 
  modifiable_indices = [literal_eval(i['modifiable_indices']) for _, i in attack_data.iterrows()] 
  modified_indices = [literal_eval(i['modified_indices']) for _, i in attack_data.iterrows()] 

  attacked_sentences = get_tokenize_sentences(premises,attacked_hypo)
  original_sentences = get_tokenize_sentences(premises,original_hypo)

  replaceable_words =[]
  replaced_words = []
  sub_words = []

  for i in range(len(original_sentences)):
    replaceable_words.append(np.array(original_sentences[i])[modifiable_indices[i]])
    sub_words.append(np.array(attacked_sentences[i])[modified_indices[i]])
    replaced_words.append(np.array(original_sentences[i])[modified_indices[i]])

  replaceable_words_freqs = []
  replaced_words_freqs = []
  sub_words_freqs = []
  sub_ovv_freqs = []

  for i in range(len(replaceable_words)):
    replaceable_words_freqs = np.concatenate((replaceable_words_freqs,get_phi_values(replaceable_words[i],freq_dict)))
    replaced_words_freqs = np.concatenate((replaced_words_freqs,get_phi_values(replaced_words[i],freq_dict)))
    sub_words_freqs = np.concatenate((sub_words_freqs,get_phi_values(sub_words[i],freq_dict)))
    sub_ovv_freqs = np.concatenate((sub_ovv_freqs,get_phi_values(sub_words[i],freq_dict, True)))

  print("MEAN:\t" + str(np.mean(replaceable_words_freqs)) + "\t STD:\t" + str(np.std(replaceable_words_freqs)))
  print("MEAN:\t" + str(np.mean(replaced_words_freqs)) + "\t STD:\t" + str(np.std(replaced_words_freqs)))
  print("MEAN:\t" + str(np.mean(sub_words_freqs)) + "\t STD:\t" + str(np.std(sub_words_freqs)))
  print("MEAN:\t" + str(np.mean(sub_ovv_freqs)) + "\t STD:\t" + str(np.std(sub_ovv_freqs)))

  #return replaceable_words_freqs,replaced_words_freqs,sub_words_freqs,sub_ovv_freqs
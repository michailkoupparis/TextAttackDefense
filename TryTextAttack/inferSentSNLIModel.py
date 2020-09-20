import torch
import textattack.models as models
import nltk

#nltk.download('punkt')
import sys

print("TRY LOADING INFERSENT FOR SNLI")

sys.modules['models'] = models

model = torch.load('../InferSent_snli_pretrained_GLOVE/InferSent_snli_pretrained_GLOVE.fullmodel') # replace this line with your model loading code
tokenizer = model.encoder # replace this line with your tokenizer loading code
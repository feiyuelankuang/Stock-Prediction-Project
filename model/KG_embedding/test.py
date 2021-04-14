import numpy as np
import csv
import os
from gensim.models import KeyedVectors
import string

model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)

def word2vec(txt):
        # Remove punctuation
    txt = txt.translate(str.maketrans('', '', string.punctuation))
        # Remove exception terms
    resultwords  = [word for word in txt.split() if word in model]
    txt = ' '.join(resultwords)
    vec = []
    for x in txt.split(' '):
        if x != '':
            print(model[x].shape)
            vec.append(model[x])
    return sum(vec)/len(vec)

def load_text(dir = '../../data/News Triple/'):
    triple = list()
    for file in os.listdir(dir):
        if file.endswith('processed.csv'):
            with open(dir + file, newline='', encoding='cp1252') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    if len(row) > 0:
                        if len(row) > 0:
                            triple.append(np.array([row[-3],row[-2],row[-1]]))
    return np.array(triple)
                       

print(load_text()[4][2])

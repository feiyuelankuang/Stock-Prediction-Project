from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import os
import string
import pickle
import tensorflow_hub as hub
import torch
from model import TranE
from nltk.tokenize import word_tokenize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)
root = '../../data/'

def word2vec(txt):
        # Remove punctuation
        #print('before process:',txt)
        #txt = txt.translate(str.maketrans('', '', string.punctuation))
        # Remove exception terms

        #resultwords  = [word for word in txt.split() if word in self.model]
        #txt = ' '.join(resultwords)
        #vec = []  
        #print('after:',txt)
      # Tokenize the string into words
    tokens = word_tokenize(txt)
       # Remove non-alphabetic tokens, such as punctuation
    words = [word.lower() for word in tokens if word.isalpha()]
       # Filter out stopwords
        #stop_words = set(stopwords.words('english'))
        #words = [word for word in words if not word in stop_words]
        #print(words)
    vector_list = [model[word] for word in words if word in model]
    if len(vector_list) == 0:
            #print('None')
        return None
    return torch.Tensor(np.mean(np.array(vector_list), axis=0)).unsqueeze(0)


def get_combined_embedding(txt,h,r,t,trans):
    h,r,t = word2vec(h),word2vec(r),word2vec(t)
    if h is None or r is None or t is None:
        return None
    kg_embedding = trans.predict(h,r,t).detach().numpy()
    sentence_embedding = embed([txt]).numpy().squeeze()
    #print(kg_embedding.shape,sentence_embedding.shape)
    return kg_embedding,np.concatenate((sentence_embedding,kg_embedding))
    
    

transE = TranE(device, d_norm=2, gamma=1).to(device)
checkpoint = torch.load('trnsE.t7')
transE.load_state_dict(checkpoint['state_dict'])

vec_def_list={}
# Loop through folder
for file in os.listdir(root+'/News Triple/'):
    if file.endswith('processed.csv'):
        news_df = pd.read_csv(root + 'News Triple/' + file, header=None, encoding='cp1252')
        #print(news_df.columns[2])

        
        #news_df = news_df.drop(news_df.columns[2], axis=1)
        #news_df = news_df.drop(news_df.columns[3], axis=1)

        news_df.rename(columns={0: "content", 1: "time",2:"score",3:"h",4:"r",5:"t"}, inplace=True)

        filename = file.split('.')[0]
        vec_df = pd.DataFrame()

        for index, row in news_df.iterrows():
            out = get_combined_embedding(row.content,row.h,row.r,row.t,transE)
            if out is not None:
                kg_embedding,combined_embedding = out
                #print('combine:',combined_embedding.shape)
                #break
            # Add to DataFrame
            vec_df = vec_df.append({"vec":combined_embedding, "time": row.time}, ignore_index=True)

        print('saveing data')
        #vec_def_list{filename} = vec_df
        vec_df.to_pickle(root + '/news_vectors/' + filename + '_combined_embedding'+'.pkl')

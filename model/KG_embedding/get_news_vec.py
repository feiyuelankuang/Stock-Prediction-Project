from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import os
import string
import pickle

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
    vector_list = [self.model[word] for word in words if word in self.model]
    if len(vector_list) == 0:
            #print('None')
        return None
    return np.mean(np.array(vector_list), axis=0)


# Loop through folder
for file in os.listdir(root+'/News Triple/'):
    if file.endswith('processed.csv'):
        news_df = pd.read_csv(root+ 'News Triple/' + file, header=None, encoding='cp1252')
        print(news_df.columns[2])

        
        #news_df = news_df.drop(news_df.columns[2], axis=1)
        #news_df = news_df.drop(news_df.columns[3], axis=1)

        news_df.rename(columns={0: "content", 1: "time",2:"score",3:"h",4:"r",5:"t"}, inplace=True)

        filename = file.split('.')[0].split('_')[:-1]
        vec_df = pd.DataFrame()

        for index, row in news_df.iterrows():
            txt = row.content
            h = row.h
            r = row.r
            t = row.t
            score = row.score
            time = row.time
            print(txt,h,r,t,score,time)
            break


            # # Remove the first 'b'
            # txt = txt[1:]

            # Remove punctuation
            txt = txt.translate(str.maketrans('', '', string.punctuation))

            # Remove exception terms
            resultwords  = [word for word in txt.split() if word in model]
            txt = ' '.join(resultwords)

            # Add to DataFrame
            vec_df = vec_df.append({"vec": [model[x] for x in txt.split(' ') if x is not ''], "time": row.time}, ignore_index=True)

        #vec_df.to_pickle('../data/news_vectors/' + filename + '.pkl')

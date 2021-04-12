from gensim.models import KeyedVectors
import pandas as pd
import os
import string
import pickle

# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

# Loop through folder
for file in os.listdir('../data/News Triple/'):
    if file.endswith('.csv'):
        news_df = pd.read_csv('../data/News Triple/' + file, header=None,encoding='cp1252')
        news_df.rename(columns={0: "content", 1: "time"}, inplace=True)

        filename = file.split('.')[0]
        vec_df = pd.DataFrame()

        for index, row in news_df.iterrows():
            txt = row.content

            # Remove the first 'b'
            #txt = txt[1:]

            # Remove punctuation
            txt = txt.translate(str.maketrans('', '', string.punctuation))

            # Remove exception terms
            resultwords  = [word for word in txt.split() if word in model]
            txt = ' '.join(resultwords)

            # Add to DataFrame
            vec_df = vec_df.append({"vec": [model[x] for x in txt.split(' ') if x is not ''], "time": row.time}, ignore_index=True)

        # vec_df.to_csv('../data/news_vectors/' + file, index=False)
        vec_df.to_pickle('../data/news_vectors/' + filename + '.pkl')
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import os
import csv
import string
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TrainSet(Dataset):
    def __init__(self, root ='../data/'):
        super(TrainSet, self).__init__()
        self.model = KeyedVectors.load_word2vec_format(root+'GoogleNews-vectors-negative300.bin', binary=True)
        self.entity_dict = {}
        self.relation_dict = {} 
        self.pos_data = []
        self.pos_vec = []
        self.neg_vec = [] 
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.load_text(root)
        self.related_dic = self.get_related_entity()         
        # print(self.related_dic[0], self.related_dic[479])
        self.generate_neg()
 

    def __len__(self):
        return self.pos_vec.shape[0]

    def __getitem__(self, item):
        #print(self.neg_data[item])
        return self.pos_vec[item],self.neg_vec[item]

    def word2vec(self, txt):
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

    def load_text(self,root):
        triple = list()
        vec = list()
        for file in os.listdir(root + '/News Triple/'):
            if file.endswith('processed.csv'):
                with open(root +'/News Triple/'+ file, newline='', encoding='cp1252') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    for row in spamreader:
                        if len(row) > 0:
                            triple = [row[-3],row[-2],row[-1]]
                            h,r,t=self.word2vec(triple[0]),self.word2vec(triple[1]),self.word2vec(triple[2])
                            if h is not None:
                                if triple[0] not in self.entity_dict:
                                    self.entity_dict[triple[0]] = h
                                if t is not None:
                                    if triple[2] not in self.entity_dict:
                                        self.entity_dict[triple[2]] = t
                                    if r is not None:
                                        if triple[1] not in self.relation_dict:
                                            self.relation_dict[triple[1]] = r
                                        self.pos_data.append(triple)
                                        self.pos_vec.append([h,r,t])
        self.pos_data = np.array(self.pos_data)
        self.pos_vec = np.array(self.pos_vec)
        
    def generate_neg(self):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        for idx, triple in enumerate(self.pos_data):
            while True:
                #print(self.pos_data.shape[0]) 
                text = random.choice(list(self.entity_dict.keys()))
                if random.randint(0, 1) == 0:
                    # replace head
                    if text not in self.related_dic[triple[2]]:
                        self.neg_vec.append([self.entity_dict[text],self.relation_dict[triple[1]],self.entity_dict[triple[2]]])
                        #print('break')
                        break
                else:
                    # replace tail
                    if text not in self.related_dic[triple[0]]:
                        self.neg_vec.append([self.entity_dict[triple[0]], self.relation_dict[triple[1]],self.entity_dict[text]])
                        #print('break')
                        break
        self.neg_vec = np.array(self.neg_vec)

    def get_related_entity(self):
        """
        get related entities
        :return: {entity_id: {related_entity_id_1, related_entity_id_2...}}
        """
        related_dic = dict()
        for triple in self.pos_data:
            if related_dic.get(triple[0]) is None:
                related_dic[triple[0]] = {triple[2]}
            else:
                related_dic[triple[0]].add(triple[2])
            if related_dic.get(triple[2]) is None:
                related_dic[triple[2]] = {triple[0]}
            else:
                related_dic[triple[2]].add(triple[0])
        return related_dic


if __name__ == '__main__':
    train_data_set = TrainSet()
    train_loader = DataLoader(train_data_set, batch_size=1, shuffle=True)
    print('yes')
    for batch_idx, (pos, neg) in enumerate(train_loader):
        print(pos.size(), neg.size())
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import os
import csv
import string

class TrainSet(Dataset):
    def __init__(self, root ='../../data'):
        super(TrainSet, self).__init__()
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.pos_data = self.load_text(root)
        self.related_dic = self.get_related_entity()
        
        # print(self.related_dic[0], self.related_dic[479])
        self.neg_data = self.generate_neg()
 

    def __len__(self):
        return self.pos_data.shape[0]

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_text(self,root):
        triple = list()
        for file in os.listdir(root + '/News Triple/'):
            if file.endswith('processed.csv'):
                with open(root +'/News Triple/'+ file, newline='', encoding='cp1252') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    for row in spamreader:
                        if len(row) > 0:
                            triple.append(np.array([row[-3],row[-2],row[-1]]))
        return np.array(triple)

    def generate_neg(self):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        neg_candidates, i = [], 0
        neg_data = []
        population = list(range(self.pos_data.shape[0]))
        for idx, triple in enumerate(self.pos_data):
            while True:
                #print(self.pos_data.shape[0])
                neg = random.randint(0,self.pos_data.shape[0])
                if random.randint(0, 1) == 0:
                    # replace head
                    if self.pos_data[neg][2] not in self.related_dic[triple[2]]:
                        neg_data.append([neg, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if self.pos_data[neg][0] not in self.related_dic[triple[0]]:
                        neg_data.append([triple[0], triple[1], neg])
                        break

        return np.array(neg_data)

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
    train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    for batch_idx, (pos, neg) in enumerate(train_loader):
        print(pos.size(), neg.size())
        break
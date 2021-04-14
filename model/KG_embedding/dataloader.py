import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
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

class TrainSet(Dataset):
    def __init__(self):
        super(TrainSet, self).__init__()
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.raw_data, self.entity_to_index, self.relation_to_index = self.load_text()
        self.entity_num, self.relation_num = len(self.entity_to_index), len(self.relation_to_index)
        self.triple_num = self.raw_data.shape[0]
        print(f'Train set: {self.entity_num} entities, {self.relation_num} relations, {self.triple_num} triplets.')
        self.pos_data = self.convert_word_to_index(self.raw_data)
        self.related_dic = self.get_related_entity()
        # print(self.related_dic[0], self.related_dic[479])
        self.neg_data = self.generate_neg()

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_text(self):
        raw_data = pd.read_csv('./fb15k/freebase_mtr100_mte100-train.txt', sep='\t', header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False, encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: x.strip())
        head_count = Counter(raw_data['head'])
        tail_count = Counter(raw_data['tail'])
        relation_count = Counter(raw_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(word, idx) for idx, word in enumerate(entity_list)])
        relation_dic = dict([(word, idx) for idx, word in enumerate(relation_list)])
        return raw_data.values, entity_dic, relation_dic

    def convert_word_to_index(self, data):
        index_list = np.array([
            [self.entity_to_index[triple[0]], self.relation_to_index[triple[1]], self.entity_to_index[triple[2]]] for
            triple in data])
        return index_list

    def generate_neg(self):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        neg_candidates, i = [], 0
        neg_data = []
        population = list(range(self.entity_num))
        for idx, triple in enumerate(self.pos_data):
            while True:
                if i == len(neg_candidates):
                    i = 0
                    neg_candidates = random.choices(population=population, k=int(1e4))
                neg, i = neg_candidates[i], i + 1
                if random.randint(0, 1) == 0:
                    # replace head
                    if neg not in self.related_dic[triple[2]]:
                        neg_data.append([neg, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if neg not in self.related_dic[triple[0]]:
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
    test_data_set = TestSet()
    test_data_set.convert_word_to_index(train_data_set.entity_to_index, train_data_set.relation_to_index,
                                                    test_data_set.raw_data)
    train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_set, batch_size=32, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        break
    # for batch_idx, (pos, neg) in enumerate(loader):
    #     # print(pos, neg)
    #     break
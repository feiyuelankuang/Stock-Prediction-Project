import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataloader import TrainSet
import math
from gensim.models import KeyedVectors


class TranE(nn.Module):
    def __init__(self, device, word_dim=300, d_norm=2, gamma=1,model_path = '../../data/GoogleNews-vectors-negative300.bin'):
        """
        :param entity_num: number of entities
        :param relation_num: number of relations
        :param dim: embedding dim
        :param device:
        :param d_norm: measure d(h+l, t), either L1-norm or L2-norm
        :param gamma: margin hyperparameter
        """
        super(TranE, self).__init__()
        self.word_dim = word_dim
        self.d_norm = d_norm
        self.device = device
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        #self.triple_num = 
        self.head_mapping = nn.Parameter(torch.FloatTensor(word_dim,word_dim))
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.tail_mapping = nn.Parameter(torch.FloatTensor(word_dim,word_dim))
        # l <= l / ||l||
        #relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.head_mapping)
        nn.init.xavier_uniform_(self.tail_mapping)

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

    def calculate_loss(self, pos_dis, neg_dis):
        """
        :param pos_dis: [batch_size, embed_dim]
        :param neg_dis: [batch_size, embed_dim]
        :return: triples loss: [batch_size]
        """
        distance_diff = self.gamma + torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis, p=self.d_norm,dim=1)
        return torch.sum(F.relu(distance_diff))

    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        """
        :param pos_head: [batch_size]
        :param pos_relation: [batch_size]
        :param pos_tail: [batch_size]
        :param neg_head: [batch_size]
        :param neg_relation: [batch_size]
        :param neg_tail: [batch_size]
        :return: triples loss
        """
        pos_dis = torch.mm(pos_head,self.head_mapping) + pos_relation - torch.mm(pos_tail,self.tail_mapping)
        neg_dis = torch.mm(neg_head,self.head_mapping) + neg_relation - torch.mm(neg_tail,self.tail_mapping)
        # return pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail
        return self.calculate_loss(pos_dis, neg_dis).requires_grad_()

if __name__ == '__main__':
    train_data_set = TrainSet()
    train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        print(data.shape)
        break
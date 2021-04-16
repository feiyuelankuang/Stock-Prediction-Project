import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from model import *
from dataloader import TrainSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 120
train_batch_size = 32
lr = 1e-2
momentum = 5e-4
gamma = 1
d_norm = 2
#top_k = 10


def main():
    train_dataset = TrainSet()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    transe = TranD(device, d_norm=2, gamma=1).to(device)
    optimizer = optim.Adam(transe.parameters(), lr=lr)#, momentum=momentum)
    #optimizer = optim.SGD(transe.parameters(), lr=lr, momentum=momentum)

    def decrease_learning_rate():
        #global optimizer
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5
    
    for epoch in range(num_epochs):
        # e <= e / ||e||
        #entity_norm = torch.norm(transe.entity_embedding.weight.data, dim=1, keepdim=True)
        #transe.entity_embedding.weight.data = transe.entity_embedding.weight.data / entity_norm
        total_loss = 0
        for batch_idx, (pos, neg) in enumerate(train_loader):
            pos, neg = pos.to(device), neg.to(device)
            # pos: [batch_size, 3] => [3, batch_size]
            pos = torch.transpose(pos, 0, 1)
            # pos_head, pos_relation, pos_tail: [batch_size]
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
            neg = torch.transpose(neg, 0, 1)
            # neg_head, neg_relation, neg_tail: [batch_size]
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            loss = transe(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 30 == 0:
            decrease_learning_rate()

    state = {
        'state_dict': transe.state_dict(),          
    }

    torch.save(state, 'trnsD.t7')    
    print("epoch:",epoch, "loss:" , total_loss/len(train_loader))


if __name__ == '__main__':
    #main()
    #train_dataset = TrainSet()
    #train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    transD = TranD(device, d_norm=d_norm, gamma=gamma).to(device)
    checkpoint = torch.load('trnsD.t7')
    transD.load_state_dict(checkpoint['state_dict'])
    #print(transD.predict('apple','is','company').detach().numpy().shape)
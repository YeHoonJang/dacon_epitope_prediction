import os
import pdb
import esm
import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer
from sklearn.metrics import f1_score



class ESMClassification(nn.Module):
    def __init__(self, num_labels=2, pretrained_no=1):
        super().__init__()
        self.num_labels = num_labels
        # self.model_name = "esm1v_t33_650M_UR90S_" + str(pretrained_no)

        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.classifier = nn.Linear(1280, self.num_labels)
        self.fc = nn.Linear(self.num_labels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, token_ids):
        outputs = self.model.forward(token_ids, repr_layers=[33])['representations'][33]
        outputs = outputs[:, :, :]
        logits = self.classifier(outputs)  # [128, max, 2]
        # output = self.fc(logits[:, -1, :]).view(-1)
        output = self.sigmoid(logits) # [128, max, 2]
        pdb.set_trace()
        # output = torch.topk(output[:, 0, :], 1).indices

        pdb.set_trace()
        # return SequenceClassifierOutput(logits=logits)
        return output.mean(dim=1)

# class MaskedMSELoss(torch.nn.Module):
#     def __init__(self):
#         super(MaskedMSELoss, self).__init__()
#
#     def forward(self, inputs, target, mask):
#         diff2 = (torch.flatten(inputs[:,:,1]) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
#         result = torch.sum(diff2) / torch.sum(mask)
#         if torch.sum(mask)==0:
#             return torch.sum(diff2)
#         else:
#             #print('loss:', result)
#             return result


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(torch.float32).to(device)
            # pdb.set_trace()
            output = model(x)
            # topk = torch.topk(output[:, 0, :], 1).indices.view(-1).to(torch.float32)
            # output = torch.topk(output, 1).indices.view(-1).to(torch.float32)
            # pdb.set_trace()
            loss = criterion(output, y)
            loss_value = loss.item()
            train_loss += loss_value
            # pdb.set_trace()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    return train_loss/total


@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(model, data_loader, criterion, device):
    model.eval()
    # criterion.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (x, y) in enumerate(data_loader):
            # pdb.set_trace()
            x = x.to(device)
            y = y.to(torch.float32).to(device)

            output = model(x)
            # output = torch.topk(output[:, 0, :], 1).indices.view(-1).to(torch.float32)
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss / total, y, output


class CustomDataset(Dataset):
    # 반드시 init, len, getitem 구현
    def __init__(self, data, alphabet):
        # 빈 리스트 생성 <- 데이터 저장
        self.X = data['epitope_seq'].str.split('').apply(lambda x:[alphabet.get_idx(x[i]) for i in range(1, len(x)-1)]).apply(lambda x: x[:100]).apply(lambda x: np.pad(x, (0,100-len(x)), 'constant', constant_values=(1)))  # sequence
        self.y = data['label']          # label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

_, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

# batch_converter = alphabet.get_batch_converter()

### Data Load
# train_df = pd.read_csv('/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/train.csv')
# test_df = pd.read_csv('/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/test.csv')

# train_data = list(train_df[['id','epitope_seq']].itertuples(index=False, name=None))
# test_data = list(test_df[['id','epitope_seq']].itertuples(index=False, name=None))
train_file = '/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/train_data.csv'
# test_file = '/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/test_data.pkl'

data = pd.read_csv(train_file)

train_data = data.iloc[:int(len(data)*0.8), :].reset_index(drop=True)
valid_data = data.iloc[int(len(data)*0.8):, :].reset_index(drop=True)


# with open(test_file, 'rb') as handle:
#     test_data = pickle.load(handle)
# pdb.set_trace()

train_data = CustomDataset(train_data, alphabet)
valid_data = CustomDataset(valid_data, alphabet)

# train_batch_labels, train_batch_strs, train_batch_tokens = batch_converter(train_data)
# test_batch_labels, test_batch_strs, test_batch_tokens = batch_converter(test_data)
# pdb.set_trace()

batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=2)    # drop_last = drop the last incomplete batch
valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=2)

model = ESMClassification().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

start_epoch = 0
epochs = 10
print("Start Training...!")
for epoch in range(start_epoch, epochs):
    print(f"Epoch: {epoch}")

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    # pdb.set_trace()

    valid_loss, label, pred = evaluate(model, valid_loader, criterion, device)
    # pdb.set_trace()
    valid_f1 = f1_score(label.cpu(), pred.cpu(), average='macro')
    print(f"Training Loss: {train_loss:.5f} | Validation Loss: {valid_loss:.5f} | F1 Score: {valid_f1:.5f}\n")

chkpt_path = '/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/checkpoint/esm_test.chkpt'
torch.save(model.state_dict(), chkpt_path)



# # with torch.no_grad():
# results = model(train_batch_tokens, repr_layers=[33], return_contacts=True).cuda()
# token_representations = results["representations"][33]
# pdb.set_trace()
#
# # Generate per-sequence representations via averaging
# # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence_representations = []
# for i, (_, seq) in enumerate(train_data):
#     sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
# pdb.set_trace()
#
# # Look at the unsupervised self-attention map contact predictions
#
# for (_, seq), attention_contacts in zip(test_data, results["contacts"]):
#     plt.matshow(attention_contacts[: len(seq), : len(seq)])
#     plt.title(seq)
#     plt.show()
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer

import esm

import pdb
import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt


class ESMClassification(nn.Module):
    def __init__(self, num_labels=2, pretrained_no=1):
        super().__init__()
        self.num_labels = num_labels
        # self.model_name = "esm1v_t33_650M_UR90S_" + str(pretrained_no)

        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.classifier = nn.Linear(1280, self.num_labels)

    def forward(self, token_ids):
        outputs = self.model.forward(token_ids, repr_layers=[33])['representations'][33]
        outputs = outputs[:, 1:-1, :]
        logits = self.classifier(outputs)

        return SequenceClassifierOutput(logits=logits)


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, target, mask):
        diff2 = (torch.flatten(inputs[:,:,1]) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        if torch.sum(mask)==0:
            return torch.sum(diff2)
        else:
            #print('loss:', result)
            return result


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss_value = loss.item()
            train_loss += loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    return train_loss/total


@torch.no_grad()    #no autograd (backpropagation X)
def evaluate(model, data_loader, criterion, device):
    model.eval()
    criterion.eval()

    valid_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            pbar.update(1)

    return valid_loss / total


class CustomDataset(Dataset):
    # 반드시 init, len, getitem 구현
    def __init__(self, data):
        # 빈 리스트 생성 <- 데이터 저장
        # TODO 여기부터
        self.X = data['epitope_seq']    # sequence
        self.y = data['label']          # label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

# Load ESM-1b model

# batch_converter = alphabet.get_batch_converter()

### Data Load
# train_df = pd.read_csv('/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/train.csv')
# test_df = pd.read_csv('/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/test.csv')

# train_data = list(train_df[['id','epitope_seq']].itertuples(index=False, name=None))
# test_data = list(test_df[['id','epitope_seq']].itertuples(index=False, name=None))
train_file = '/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/train_data.csv'
# test_file = '/home/yehoon/workspace/dacon_epitope_prediction/SEMAi/data/test_data.pkl'

data = pd.read_csv(train_file)

train_data = data.iloc[:int(len(data)*0.8), :]
valid_data = data.iloc[int(len(data)*0.8):, :]


# with open(test_file, 'rb') as handle:
#     test_data = pickle.load(handle)
pdb.set_trace()

# train_batch_labels, train_batch_strs, train_batch_tokens = batch_converter(train_data)
# test_batch_labels, test_batch_strs, test_batch_tokens = batch_converter(test_data)
pdb.set_trace()

model = ESMClassification()
_, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
pdb.set_trace()

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=8)    # drop_last = drop the last incomplete batch
valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=8)


# with torch.no_grad():
results = model(train_batch_tokens, repr_layers=[33], return_contacts=True).cuda()
token_representations = results["representations"][33]
pdb.set_trace()

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, (_, seq) in enumerate(train_data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
pdb.set_trace()

# Look at the unsupervised self-attention map contact predictions

for (_, seq), attention_contacts in zip(test_data, results["contacts"]):
    plt.matshow(attention_contacts[: len(seq), : len(seq)])
    plt.title(seq)
    plt.show()
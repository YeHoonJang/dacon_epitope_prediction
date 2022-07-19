import os
# set cuda params
# 'TORCH_HOME'directory will be used to save origenal esm-1v weights
os.environ['TORCH_HOME'] = "../torch_hub"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import scipy
import sklearn
import esm

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch import nn
import math

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments, EvalPrediction

from esm.pretrained import load_model_and_alphabet_hub

from sklearn.metrics import r2_score, mean_squared_error


class PDB_Dataset(Dataset):
    """
    A class to represent a sutable data set for model.

    convert original pandas data frame to model set,
    where 'token ids' is ESM-1v embedings corresponed to protein sequence (max length 1022 AA)
    and 'lables' is a contact number values
    Attributes:
        df (pandas.DataFrame): dataframe with two columns:
                0 -- preotein sequence in string ('GLVM') or list (['G', 'L', 'V', 'M']) format
                1 -- contcat number values in list [0, 0.123, 0.23, -100, 1.34] format
        esm1v_batch_converter (function):
                    ESM function callable to convert an unprocessed (labels + strings) batch to a
                    processed (labels + tensor) batch.
        label_type (str):
                type of model: regression or binary

    """

    def __init__(self, df, label_type='regression'):
        """
        Construct all the necessary attributes to the PDB_Database object.

        Parameters:
            df (pandas.DataFrame): dataframe with two columns:
                0 -- protein sequence in string ('GLVM') or list (['G', 'L', 'V', 'M']) format
                1 -- contcat number values in list [0, 0.123, 0.23, -100, 1.34] format
            label_type (str):
                type of model: regression or binary
        """
        self.df = df
        _, esm1v_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        # _, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
        # _, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_1")
        self.esm1v_batch_converter = esm1v_alphabet.get_batch_converter()
        self.label_type = label_type

    def __getitem__(self, idx):
        item = {}
        _, _, esm1b_batch_tokens = self.esm1v_batch_converter([('', ''.join(self.df.iloc[idx, 0])[:1022])])
        item['token_ids'] = esm1b_batch_tokens
        item['labels'] = torch.unsqueeze(torch.LongTensor(self.df.iloc[idx, 1][:1022]), 0)

        return item

    def __len__(self):
        return len(self.df)


class ESM1vForTokenClassification(nn.Module):

    def __init__(self, num_labels=2, pretrained_no=1):
        super().__init__()
        self.num_labels = num_labels
        # self.model_name = "esm1v_t33_650M_UR90S_" + str(pretrained_no)
        self.model_name = "esm1b_t33_650M_UR50S" + str(pretrained_no)


        self.esm1v, self.esm1v_alphabet = load_model_and_alphabet_hub(self.model_name)
        self.classifier = nn.Linear(1280, self.num_labels)

    def forward(self, token_ids):
        outputs = self.esm1v.forward(token_ids, repr_layers=[33])['representations'][33]
        outputs = outputs[:, 1:-1, :]
        logits = self.classifier(outputs)

        return SequenceClassifierOutput(logits=logits)


def compute_metrics_regr(p: EvalPrediction):
    preds = p.predictions[:, :, 1]

    batch_size, seq_len = preds.shape
    out_labels, out_preds = [], []

    for i in range(batch_size):
        for j in range(seq_len):
            if p.label_ids[i, j] > -1:
                out_labels.append(p.label_ids[i][j])
                out_preds.append(preds[i][j])

    out_labels_regr = [math.log(t + 1) for t in out_labels]

    return {
        "pearson_r": scipy.stats.pearsonr(out_labels_regr, out_preds)[0],
        "mse": mean_squared_error(out_labels_regr, out_preds),
        "r2_score": r2_score(out_labels_regr, out_preds)
    }


## you can train one model or ensemple of 5 models
def model_init_1():
    return ESM1vForTokenClassification(pretrained_no = 1).cuda()
def model_init_2():
    return ESM1vForTokenClassification(pretrained_no = 2).cuda()
def model_init_3():
    return ESM1vForTokenClassification(pretrained_no = 3).cuda()
def model_init_4():
    return ESM1vForTokenClassification(pretrained_no = 4).cuda()
def model_init_5():
    return ESM1vForTokenClassification(pretrained_no = 5).cuda()


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, target, mask):
        diff2 = (torch.flatten(inputs[:, :, 1]) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        if torch.sum(mask) == 0:
            return torch.sum(diff2)
        else:
            # print('loss:', result)
            return result


class MaskedRegressTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.squeeze().detach().cpu().numpy().tolist()
        labels = [math.log(t + 1) if t != -100 else -100 for t in labels]
        labels = torch.unsqueeze(torch.FloatTensor(labels), 0).cuda()
        masks = ~torch.eq(labels, -100).cuda()

        # masks = inputs.pop("masks")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = MaskedMSELoss()

        loss = loss_fn(logits, labels, masks)

        return (loss, outputs) if return_outputs else loss

def collator_fn(x):
    if len(x)==1:
        return x[0]
    print('x:', x)
    return x


train_set = pd.read_csv('../data/train_set.csv')
train_set = train_set.groupby('pdb_id_chain').agg({'resi_pos': list,
                                 'resi_aa': list,
                                 'contact_number': list}).reset_index()
## the first run will ake about 5-10 minutes, because esm weights should be downloaded
train_ds = PDB_Dataset(train_set[['resi_aa', 'contact_number']],
                      label_type ='regression')

test_set = pd.read_csv('../data/test_set.csv')
test_set = test_set.groupby('pdb_id_chain').agg({'resi_pos': list,
                                 'resi_aa': list,
                                 'contact_number': list}).reset_index()
test_ds = PDB_Dataset(test_set[['resi_aa', 'contact_number']],
                      label_type ='regression')

training_args = TrainingArguments(
    output_dir='./results_fold' ,          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    learning_rate=1e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    save_strategy = "no",
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=1,  # total number of steps before back propagation
    fp16=False,                       # Use mixed precision
    run_name="PDB_binary",      # experiment name
    seed=42,                         # Seed for experiment reproducibility
    load_best_model_at_end=False,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

)

#create direactory to weights storage
if not os.path.exists("../models/"):
    os.makedirs("../models/")


trainer = MaskedRegressTrainer(
    model=model_init_1(),                   # the instantiated 🤗 Transformers model to be trained
    args=training_args,                     # training arguments, defined above
    train_dataset=train_ds,                 # training dataset
    eval_dataset=test_ds,                   # evaluation dataset
    data_collator = collator_fn,
    compute_metrics = compute_metrics_regr,    # evaluation metrics
)

trainer.train()

#save weights
torch.save(trainer.model.state_dict(), "../models/sema_1d_0.pth")

for idx, model_init in enumerate([model_init_1, model_init_2, model_init_3, model_init_4, model_init_5]):
    trainer = MaskedRegressTrainer(
        model=model_init(),  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=test_ds,  # evaluation dataset
        data_collator=collator_fn,
        compute_metrics=compute_metrics_regr  # evaluation metrics
    )

    trainer.train()

    # save weights
    torch.save(trainer.model.state_dict(), f"../models/sema_1d_{str(idx)}.pth")
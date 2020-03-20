import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Function
from sklearn.model_selection import *
from transformers import *
from tqdm import tqdm

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain_model_path = '../bert'
max_seq_len = [200,250,250]

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNmodel(nn.Module):
    def __init__(self):
        super(DANNmodel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)

        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)
        self.discriminator = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, alpha=1):
        outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)

        feature = outputs[1]

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output

class MyDataset(Dataset):
    def __init__(self, df, mode='train', task=9):
        self.mode = mode
        self.task = task
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.x_data = []
        self.y_data = []
        for i, row in tqdm(df.iterrows()):
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def contact(self, str1, str2):
        if pd.isnull(str2):
            return str1
        return str1 + str2
    def row_to_tensor(self, tokenizer, row):
        if self.task == 0:
            text = row['content']
        elif self.task == 1:
            text = self.contact(row['content'], row['comment_2c'])
        else:
            text = self.contact(row['content'], row['comment_all'])
        x_encode = tokenizer.encode(text)
        if len(x_encode) > max_seq_len[self.task]:
            text_len = int(max_seq_len[self.task] / 2)
            x_encode = x_encode[:text_len] + x_encode[-text_len:]
        else:
            padding = [0] * (max_seq_len[self.task] - len(x_encode))
            x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * 3, dtype=torch.long)
        else:
            # y_data = row[config.label_columns]
            y_data = row['label']
            y_tensor = torch.tensor(y_data, dtype=torch.long)
        return x_tensor, y_tensor
    def __len__(self):
        return len(self.y_data)

model = DANNmodel().to(device)

def train(train_data, val_data, tgt, epochs=5, batch_size=4):
    train_dataset = MyDataset(train_data, 'train', 2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(val_data, 'val', 2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    tgt_dataset = MyDataset(tgt, 'test', 2)
    tgt_loader = DataLoader(dataset=tgt_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        print('epoch:{}'.format(epoch + 1))

        model.train()

        len_dataloader = min(len(train_loader), len(tgt_loader))
        data_zip = enumerate(zip(train_loader, tgt_loader))

        for step, ((text_src, class_src), (text_tgt, _)) in data_zip:
            label_src = torch.zeros(len(text_src)).long().to(device)
            label_tgt = torch.ones(len(text_tgt)).long().to(device)
            class_src = class_src.to(device)
            text_src = text_src.to(device)
            text_tgt = text_tgt.to(device)

            p = float((step + epoch * len_dataloader) / (epochs * len_dataloader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            src_class_output, src_domain_output = model(text_src,alpha=alpha)
            src_loss_class = criterion(src_class_output, class_src)
            src_loss_domain = criterion(src_domain_output, label_src)

            _, tgt_domain_output = model(text_tgt, alpha=alpha)
            tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

            loss = src_loss_class + src_loss_domain + tgt_loss_domain

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                msg = 'step: {0}/{1}, train loss: {2}'
                print(msg.format(step, len_dataloader, loss.item()),src_loss_class.item(),src_loss_domain.item(),tgt_loss_domain.item())
        torch.save(model.state_dict(), './model.bin')

train_df = pd.read_csv('./train.csv')
train_df.dropna(subset=['content'], inplace=True)
train_df['label'] = train_df[['ncw_label','fake_label','real_label']].values.argmax(axis=1)

test_df = pd.read_csv('./test.csv')
test_df['content']=test_df['content'].fillna(value='')


train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
print('train:{}, val:{}'.format(train_data.shape, val_data.shape))

# model.load_state_dict(torch.load('./model.bin'))
train(train_data, val_data, test_df)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import gc

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AdamW

from src.dataset import MyDataset
from src.model import MyModel

from run import *


config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = MyModel(config)
criterion = nn.MarginRankingLoss(margin=0.5)
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scheduler = None

#class MyDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length)
train_data,test_data = read_data()
tmp_train_data = train_data.sample(frac=0.01,random_state=200)
train_loader = DataLoader(MyDataset(tmp_train_data,tokenizer,config.max_length),batch_size=16,shuffle=True)
test_loader = DataLoader(MyDataset(tmp_train_data,tokenizer,config.max_length),batch_size=16)

model.to(device)

for i in range(10):
    train_one_epoch(model,config,optimizer,criterion,scheduler,train_loader,device,i)
    test_one_epoch(model,criterion,test_loader,device,i)

# right = 0
# total = 0
# for step, data in enumerate(train_loader):
#     more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
#     more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
#     less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
#     less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
#     targets = data['target'].to(device, dtype=torch.long)
        
#     batch_size = more_toxic_ids.size(0)
#     print(batch_size)

#     more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
#     print(more_toxic_outputs)
#     less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
#     print(less_toxic_outputs)
#     right += (more_toxic_outputs>less_toxic_outputs).sum()
#     print(right)
#     total += batch_size 
        
#     loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
#     print(loss)
#     break
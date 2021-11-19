
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


def read_data():
    df = pd.read_csv("data/validation_data.csv")

    #训练集和测试集分割
    train_data = df.sample(frac=0.9,random_state=200) #random state is a seed value
    test_data = df.drop(train_data.index)
    print(train_data.shape)
    print(test_data.shape)
    return train_data,test_data

def train_one_epoch(model,config, optimizer, criterion,scheduler, dataloader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
        loss = loss / config.n_accumulate
        loss.backward()
    
        if (step + 1) % config.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss


@torch.no_grad()
def test_one_epoch(model,criterion,dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    right = 0
    total = 0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        right += (more_toxic_outputs>less_toxic_outputs).sum()
        total += batch_size 
        
        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
        
    print("acc:{}".format(right/total))
    
    gc.collect()
    
    return epoch_loss


from dataclasses import dataclass

@dataclass
class Config:
    """train config"""
    model_name: str = "bert-base-cased"
    hidden_dim: int = 768
    max_length: int = 512
    dropout: float = 0.3
    num_labels: int = 1
    lr: float = 0.001
    epoch: int = 2
    n_accumulate: int = 1
    weight_decay: float = 1e-6


#train!
if __name__ == "__main__":
    # def train_one_epoch(model, config,optimizer,criterion, scheduler, dataloader, device, epoch)
    # def test_one_epoch(model, criterion,dataloader, device, epoch):
    train_data,test_data = read_data()
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MyModel(config)
    criterion = nn.MarginRankingLoss(margin=0.5)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scheduler = None

    #class MyDataset(Dataset):
    #     def __init__(self, df, tokenizer, max_length)
    tmp_train_data = train_data.sample(frac=0.01,random_state=200)
    train_loader = DataLoader(MyDataset(tmp_train_data,tokenizer,config.max_length),batch_size=16,shuffle=True)
    test_loader = DataLoader(MyDataset(tmp_train_data,tokenizer,config.max_length),batch_size=16)

    model.to(device)

    for i in range(config.epoch):
        train_one_epoch(model,config,optimizer,criterion,scheduler,train_loader,device,i)
        test_one_epoch(model,criterion,test_loader,device,i)
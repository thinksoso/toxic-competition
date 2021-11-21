import wandb

from dataclasses import dataclass
import copy
import pdb
import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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

from src.process import preprocess


def read_data(debug):
    df = pd.read_csv("data/validation_data.csv")
    df = preprocess(df)

    # 训练集和测试集分割
    # random state is a seed value
    train_data = df.sample(frac=0.9, random_state=200)
    test_data = df.drop(train_data.index)

    # 开启debug模式，只会取少量数据测试
    if debug:
        train_data = train_data.sample(frac=0.01, random_state=200)
        test_data = test_data.sample(frac=0.01, random_state=200)
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data


def save_model(model, epoch, score, config):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath = config.model_save_path+"{}-{}-epoch{}-score{}.pt".format(
        timestr, config.model_name, epoch, score)
    torch.save(model.state_dict(), filepath)


def train_one_epoch(model, config, optimizer, criterion, dataloader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype=torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype=torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype=torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype=torch.long)
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
        
        wandb.log({"loss":loss.item()})

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def test_one_epoch(model, criterion, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    right = 0
    total = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype=torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype=torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype=torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = more_toxic_ids.size(0)
        # pdb.set_trace()
        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        right += (more_toxic_outputs > less_toxic_outputs).sum()
        total += batch_size

        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    acc = right/total

    print("acc:{}".format(acc))

    gc.collect()

    return acc


def AdamW_LLRD(model):
    opt_parameters = []

    init_lr = 5e-6
    head_lr = 5e-6
    fc_lr = 1e-5
    lr = init_lr

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    named_parameters = list(model.named_parameters())

    # pooler
    params_0 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}
    opt_parameters.append(head_params)

    # 12 hidden layers
    for layer in range(11, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)

        lr *= 0.9

    # embedding
    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
    opt_parameters.append(embed_params)

    # 加的全连接层
    params_0 = [p for n, p in named_parameters if "fc" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "fc" in n
                and not any(nd in n for nd in no_decay)]

    fc_params = {"params": params_0, "lr": fc_lr, "weight_decay": 0.0}
    opt_parameters.append(fc_params)
    fc_params = {"params": params_1, "lr": fc_lr, "weight_decay": 0.01}
    opt_parameters.append(fc_params)

    return AdamW(opt_parameters, lr=init_lr)


@dataclass
class Config:
    """train config"""
    model_name: str = "roberta-base"
    hidden_dim: int = 768
    max_length: int = 512
    dropout: float = 0.5
    num_labels: int = 1
    lr: float = 0.000003
    epoch: int = 16
    n_accumulate: int = 1
    weight_decay: float = 0.01
    model_save_path: str = "./model_saved/"
    debug: bool = True


# train!
if __name__ == "__main__":
    # def train_one_epoch(model, config,optimizer,criterion, scheduler, dataloader, device, epoch)
    # def test_one_epoch(model, criterion,dataloader, device, epoch):
    config = Config()
    train_data, test_data = read_data(config.debug)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MyModel(config)
    criterion = nn.MarginRankingLoss(margin=0.5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # wandb
    wandb.init(project="toxic",entity="thinksoso")
    wandb.config = {
        "learning_rate": 5e-6,
        "epochs":10,
        "batch_size":8
    }
    wandb.watch(model)

    # optimizer = AdamW(model.parameters(), lr=config.lr,
    #   weight_decay=config.weight_decay)
    optimizer = AdamW_LLRD(model)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, total_iters=5)

    # class MyDataset(Dataset):
    #     def __init__(self, df, tokenizer, max_length)
    train_loader = DataLoader(MyDataset(
        train_data, tokenizer, config.max_length), batch_size=8, shuffle=True)
    test_loader = DataLoader(
        MyDataset(test_data, tokenizer, config.max_length), batch_size=8)

    model.to(device)

    best_acc = 0.0
    best_model = None
    best_epoch = 0
    for i in range(config.epoch):
        train_one_epoch(model, config, optimizer, criterion,
                        train_loader, device, i)
        acc = test_one_epoch(model, criterion, test_loader, device, i)

        wandb.log({"loss":acc})

        if best_acc <= acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            best_epoch = i

        if i % 5 == 0 and i != 0:
            save_model(model, i, acc, config)

        scheduler.step()

    save_model(best_model, best_epoch, best_acc, config)

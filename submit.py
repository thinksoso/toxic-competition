import time
from torch._C import TreeView
from tqdm import tqdm
import pdb

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

import pandas as pd

from src.model import MyModel
from src.dataset import SubmitDataset
from run import Config

from transformers import AutoTokenizer, AutoModel, AdamW

def rerank(comment_ids,scores):
    _,new_comment_ids = zip(*sorted(zip(scores,comment_ids)))
    rank = {}
    for i, c in enumerate(new_comment_ids):
        rank[c] = i

    new_scores = []
    for c in comment_ids:
        new_scores.append(rank[c])
    
    return comment_ids,new_scores


@torch.no_grad()
def summit(model,data_loader,device):
    model.eval()
    comment_ids = []
    scores = []
    for i,data in enumerate(tqdm(data_loader)):
        comment_id = data["comment_id"].tolist()
        text_mask = data["inputs_text_mask"].to(device,dtype=torch.long)
        text_id = data["inputs_text_ids"].to(device,dtype=torch.long)
        
        score = model(text_id,text_mask)
        score = torch.squeeze(score,dim=1).tolist()

        scores += score
        comment_ids += comment_id
    
    comment_ids,scores = rerank(comment_ids,scores)
    df = pd.DataFrame(list(zip(comment_ids,scores)),columns=["comment_id","score"])
    return df

if __name__ == "__main__":
    modelpath = "model_saved/20211120-145022-bert-base-uncased-epoch1-score1.0.pt"
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MyModel(config)
    model.load_state_dict(torch.load(modelpath))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    df = pd.read_csv("data/comments_to_score.csv")
    dataset = SubmitDataset(df,tokenizer,config.max_length)
    dataloader = DataLoader(dataset=dataset,batch_size=16)

    result = summit(model,dataloader,device)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "submission/"+timestr+"-summit.csv"
    result.to_csv(save_path,index=False)
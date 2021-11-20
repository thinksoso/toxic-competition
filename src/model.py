import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AdamW

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.model_name)
        self.drop = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(config.hidden_dim,config.num_labels)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
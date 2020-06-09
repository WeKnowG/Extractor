import sys
sys.path.append("../")

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubJectModel(nn.Module):
    
    def __init__(self):
        
        super(SubJectModel, self).__init__()
        
        self.projection_heads = nn.Linear(config.bert_feature_size, 1)
        self.projection_tails = nn.Linear(config.bert_feature_size, 1)
        self.projection_heads.to(config.device)
        self.projection_tails.to(config.device)
        
    def forward(self, bert_outputs):
        
        sub_heads = self.projection_heads(bert_outputs)
        sub_tails = self.projection_tails(bert_outputs)
        
        b, _, _ = list(sub_heads.size())
        
        sub_heads = torch.sigmoid(sub_heads).view(b, -1)
        sub_tails = torch.sigmoid(sub_tails).view(b, -1)
        
        return sub_heads, sub_tails
        
        
        
        
        
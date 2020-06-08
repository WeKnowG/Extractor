import sys

import torch
import torch.nn as nn
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertPreTrainedModel, BertModel, BertConfig

class BERT(BertPreTrainedModel):
    
    config_class = BertConfig
    pertrained_model_archieve_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    
    def __init__(self, config):
        
        super(BERT, self).__init__(config)
        self.bert_model = BertModel(config=config)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        return outputs[0]
        
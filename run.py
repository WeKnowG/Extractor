import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loader import load_data, collate_wrapper, CustomDataset, CustomBatch
from utils.metric import metric
import config
from models import SubJectModel, ObjectModel


class Trainer:

    def __init__(self, dataset):
        
        train_path = config.dataset_path + dataset + "/train_triples.json"
        dev_path = config.dataset_path + dataset + "/dev_triples.json"
        test_path = config.dataset_path + dataset + "/test_triples.json"
        rel_dict_path = config.dataset_path + dataset + "/rel2id.json"

        # data process
        self.train_data, self.dev_data, self.test_data, self.id2rel, self.rel2id, self.num_rels = load_data(train_path, 
                                                                                                            dev_path, 
                                                                                                            test_path, 
                                                                                                            rel_dict_path)
    def setup(self, model):
        
        bert_config_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/config.json"
        bert_model_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/model.bin"
        bert_vocab_path = config.pretrained_model_path + config.MODEL_PATH_MAP[model] + "/vocab.txt"

        lm_config = config.MODEL_CLASSES[model][0].from_pretrained(bert_config_path)
        self.lm_model = config.MODEL_CLASSES[model][1].from_pretrained(bert_model_path, config=lm_config).to(config.device)
        self.lm_tokenizer = config.MODEL_CLASSES[model][2](bert_vocab_path)
         
        self.train_data = CustomDataset(self.train_data, 
                                        self.lm_tokenizer, 
                                        self.rel2id, 
                                        self.num_rels)
#         self.dev_data = CustomDataset(self.dev_data, 
#                                       lm_tokenizer, 
#                                       self.rel2id, 
#                                       self.num_rels)
#         self.test_data = CustomDataset(self.test_data,
#                                        lm_tokenizer, 
#                                        self.rel2id, 
#                                        self.num_rels)
        
        # set data loader
        self.train_batcher = DataLoader(self.train_data, 
                                        config.batch_size, 
                                        drop_last=True, 
                                        shuffle=True, 
                                        collate_fn=collate_wrapper)
#         self.dev_batcher = DataLoader(self.dev_data, 
#                                       config.batch_size,
#                                       drop_last=True, 
#                                       shuffle=False, 
#                                       collate_fn=collate_wrapper)
#         self.test_batcher = DataLoader(self.test_data, 
#                                        config.batch_size, 
#                                        drop_last=True, 
#                                        shuffle=False, 
#                                        collate_fn=collate_wrapper)
        
        self.subject_model = SubJectModel() 
        self.object_model = ObjectModel(self.num_rels)
        
        self.criterion = nn.BCELoss(reduction="none")
        
        self.models_params = list(self.lm_model.parameters()) + list(self.subject_model.parameters()) + list(self.object_model.parameters())
        
        self.optimizer = torch.optim.Adam(self.models_params, lr=config.lr)

    def trainIters(self):
        
        step = 0
        for epoch in range(config.epoches): 
            
            for batch in self.train_batcher:
                total_loss, sub_entities_loss, obj_entities_loss = self.train_one_batch(batch)
                print(epoch, total_loss, sub_entities_loss, obj_entities_loss)
                step += 1
#                 if step % 5 == 0:
#                     metric(self.lm_model, self.subject_model, self.object_model, self.test_data, self.id2rel, self.lm_tokenizer, output_path="./result.json")
                    
#                     print(batch.sub_heads.cpu().numpy().tolist()[0])
                    
#                     print(heads.cpu().detach().numpy().tolist()[0])
                    
#                     print(batch.sub_tails.cpu().numpy().tolist()[0])
#                     print(tails.cpu().detach().numpy().tolist()[0])
#                     print("-"*20)
                    
#                     for i in range(len(batch.sub_heads.cpu().numpy().tolist()[0])):
#                         if batch.sub_heads.cpu().numpy().tolist()[0][i] == 1:
#                             print(heads.cpu().detach().numpy().tolist()[0][i])
#                     for i in range(len(batch.sub_tails.cpu().numpy().tolist()[0])):
#                         if batch.sub_tails.cpu().numpy().tolist()[0][i] == 1:
#                             print(tails.cpu().detach().numpy().tolist()[0][i])
#                     print("-"*20)
            
            self.reset_train_dataloader()
    
    def reset_train_dataloader(self):
        
        self.train_batcher = DataLoader(self.train_data, 
                                        config.batch_size, 
                                        drop_last=True, 
                                        shuffle=True, 
                                        collate_fn=collate_wrapper)

    def train_one_batch(self, batch):
        
        self.optimizer.zero_grad()
        tokens_batch = batch.token_ids
        attention_mask_batch = batch.attention_mask
        segments_batch = batch.segment_ids
        sub_heads_batch = batch.sub_heads
        sub_tails_batch = batch.sub_tails
        sub_head_batch = batch.sub_head
        sub_tail_batch = batch.sub_tail
        
        obj_heads_batch = batch.obj_heads
        obj_tails_batch = batch.obj_tails
            
        bert_inputs = {"input_ids": tokens_batch, 
                       "attention_mask": attention_mask_batch, 
                       "token_type_ids": segments_batch}
        
        bert_outputs = self.lm_model(**bert_inputs)
        
        sub_heads, sub_tails = self.subject_model(bert_outputs)
        
        pred_obj_heads, pred_obj_tails = self.object_model(bert_outputs, 
                                                           sub_head_batch, 
                                                           sub_tail_batch)
        
        
        sub_heads_loss = self.criterion(sub_heads, sub_heads_batch)
        sub_tails_loss = self.criterion(sub_tails, sub_tails_batch)
    
        obj_heads_loss = self.criterion(pred_obj_heads, obj_heads_batch)
        obj_tails_loss = self.criterion(pred_obj_tails, obj_tails_batch)
        
        sub_entities_loss = ((sub_heads_loss + sub_tails_loss) * attention_mask_batch).sum() / attention_mask_batch.sum()
        
        obj_entities_loss = ((obj_heads_loss + obj_tails_loss) * attention_mask_batch.unsqueeze(-1)).sum() / attention_mask_batch.unsqueeze(-1).sum()
        
        
        total_loss = sub_entities_loss + obj_entities_loss
        
        total_loss.backward()
        
        self.optimizer.step()
        
        return total_loss.item(), sub_entities_loss.item(), obj_entities_loss.item()
           
        
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument("--model", default="bert", type=str, help="specify the type of language models")
    parser.add_argument("--dataset", default="NYT", type=str, help="specify the dataset")
    args = parser.parse_args()
    trainer = Trainer(args.dataset)
    trainer.setup(args.model)
    trainer.trainIters()
import os
import json
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loader import load_data, collate_wrapper, CustomDataset, CustomBatch
from utils.metric import metric
import config
from models import SubJectModel, ObjectModel


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
        self.lm_model = nn.DataParallel(config.MODEL_CLASSES[model][1].from_pretrained(bert_model_path, config=lm_config)).to(config.device)
        self.lm_tokenizer = config.MODEL_CLASSES[model][2](bert_vocab_path, do_lower_case=False)
         
        self.train_data = CustomDataset(self.train_data, 
                                        self.lm_tokenizer, 
                                        self.rel2id, 
                                        self.num_rels)
        
        # set data loader
        self.train_batcher = DataLoader(self.train_data, 
                                        config.batch_size, 
                                        drop_last=True, 
                                        shuffle=True, 
                                        collate_fn=collate_wrapper)
        
        self.subject_model = nn.DataParallel(SubJectModel()).to(config.device) 
        self.object_model = nn.DataParallel(ObjectModel(self.num_rels)).to(config.device)
        
        self.criterion = nn.BCELoss(reduction="none")
        
        self.models_params = list(self.lm_model.parameters()) + list(self.subject_model.parameters()) + list(self.object_model.parameters())
        
        self.optimizer = torch.optim.Adam(self.models_params, lr=config.lr)
        
        self.start_step = None
        
        if config.load_weight:
            print("start loading weight...")
            
            state = torch.load(config.model_file_path, map_location= lambda storage, location: storage)
            self.lm_model.module.load_state_dict(state['lm_model'])
            self.object_model.module.load_state_dict(state['object_model'])
            self.subject_model.module.load_state_dict(state['subject_model'])
            
            self.start_step = state['step']
            
            self.optimizer.load_state_dict(state['optimizer'])
            if config.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            
        
    def save_models(self, args, total_loss, step):
        
        state = {
            "step": step,
            "lm_model": self.lm_model.module.state_dict(),
            "object_model": self.object_model.module.state_dict(),
            "subject_model": self.subject_model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_loss": total_loss
        }
        model_save_path = os.path.join(config.save_weights_path, "%s_model_%d_%d" % (args.model, step, int(time.time())) )
        torch.save(state, model_save_path)
        
        

    def trainIters(self, args):
        
        self.lm_model.train()
        self.subject_model.train()
        self.object_model.train()
        
        if self.start_step:
            step = self.start_step
        else:
            step = 0
            
        for epoch in range(config.epoches): 
            
            for batch in self.train_batcher:
                total_loss, sub_entities_loss, obj_entities_loss = self.train_one_batch(batch)
                print("epoch:", epoch, "step: ", step, "total_loss:", total_loss, "sub_entities_loss:", sub_entities_loss, "obj_entities_loss: ", obj_entities_loss)
                step += 1
                if step % 2000 == 0:
                    
                    with torch.no_grad():
                        
                        self.lm_model.eval()
                        self.subject_model.eval()
                        self.object_model.eval()

                        precision, recall, f1 = metric(self.lm_model, self.subject_model, self.object_model, self.test_data, self.id2rel, self.lm_tokenizer, output_path="./result.json")
                        print("precision: ", precision, "recall: ", recall, "f1: ", f1)
                        self.save_models(args, total_loss, step)
                        
                    self.lm_model.train()
                    self.subject_model.train()
                    self.object_model.train()
            
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
        
        if config.focal_loss:
        
            sub_entities_loss = ((torch.abs(sub_heads_batch - sub_heads) * sub_heads_loss + 
                                  torch.abs(sub_tails_batch - sub_tails) * sub_tails_loss) * attention_mask_batch).sum() / attention_mask_batch.sum()

            obj_entities_loss = ((torch.abs(obj_heads_batch - pred_obj_heads) * obj_heads_loss + 
                                  torch.abs(obj_tails_batch - pred_obj_tails) * obj_tails_loss) * attention_mask_batch.unsqueeze(-1)).sum() / attention_mask_batch.unsqueeze(-1).sum()
        else:
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
    trainer.trainIters(args)
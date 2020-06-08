import sys
sys.path.append("../")

import numpy as np
import re
import json
from random import choice
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import config



def find_head_idx(source, target):
    
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def to_tuple(sent):
    
    triple_list = []
    for triple in sent["triple_list"]:
        triple_list.append(tuple(triple))
    sent["triple_list"] = triple_list
    
def seq_padding(seq, padding=0):
    
    return np.concatenate([seq, [padding] * (config.BERT_MAX_LEN - len(seq))]) if len(seq) < config.BERT_MAX_LEN else seq

def load_data(train_path, dev_path, test_path, rel_dict_path):
    
    train_data = json.load(open(train_path))
    dev_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2rel, rel2id = json.load(open(rel_dict_path))
    
    id2rel = {int(i): j for i, j in id2rel.items()}
    
    num_rels = len(id2rel)
    
    random_order = list(range(len(train_data)))
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(random_order)
    train_data = [train_data[i] for i in random_order]
    
    for sent in train_data:
        to_tuple(sent)
    for sent in dev_data:
        to_tuple(sent)
    for sent in test_data:
        to_tuple(sent)
    
    print("train data len:", len(train_data))
    print("dev data len:", len(dev_data))
    print("test data len:", len(test_data))
    
    return train_data, dev_data, test_data, id2rel, rel2id, num_rels

class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, rel2id, num_rels):
        
        self.data = data
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_rels = num_rels
        
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.unk_token = tokenizer.unk_token
        self.pad_token_id = tokenizer.pad_token_id
        
        self.sequence_a_segment_id = 0
        self.cls_token_segment_id = 0
        
        self.get_examples()

    def get_examples(self):
        
        examples = []
        
        for idx in tqdm(range(len(self.data))):
            
            line = self.data[idx]
            text = " ".join(line["text"].split()[:config.max_text_len])
            tokens = [self.cls_token] + self.tokenizer.tokenize(text) + [self.sep_token]
            
            if len(tokens) > config.BERT_MAX_LEN:
                tokens = tokens[:config.BERT_MAX_LEN]
            text_len = len(tokens)
            
            s2ro_map = {}
            for triple in line["triple_list"]:
                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, 
                                          obj_head_idx + len(triple[2]) - 1, 
                                          self.rel2id[triple[1]]))
            
            if s2ro_map:
                
                example = {}
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                attention_mask = [1] * len(token_ids)                
                segment_ids = [self.cls_token_segment_id] + ([self.sequence_a_segment_id] * text_len)[1:]
                assert len(token_ids) == text_len, "Error !"
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head, sub_tail = choice(list(s2ro_map.keys()))
                obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                for ro in s2ro_map.get((sub_head, sub_tail), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                
                token_ids = seq_padding(token_ids)
                attention_mask = seq_padding(attention_mask)
                segment_ids = seq_padding(segment_ids)
                sub_heads = seq_padding(sub_heads)
                sub_tails = seq_padding(sub_tails)
                obj_heads = seq_padding(obj_heads, np.zeros(self.num_rels))
                obj_tails = seq_padding(obj_tails, np.zeros(self.num_rels))
                
                example["token_ids"] = token_ids
                example["attention_mask"] = attention_mask
                example["segment_ids"] = segment_ids
                example["sub_heads"] = sub_heads
                example["sub_tails"] = sub_tails
                example["obj_heads"] = obj_heads
                example["obj_tails"] = obj_tails
                example["sub_head"] = np.array(sub_head)
                example["sub_tail"] = np.array(sub_tail)  
                example["text"] = text
                example["tokens"] = tokens
                
                examples.append(example)
                
        self.examples = examples        
        self.data_len = len(self.examples)
                
        
    def __getitem__(self, index):
        
        example = self.examples[index]
        
        token_ids = example["token_ids"]
        attention_mask = example["attention_mask"]
        segment_ids = example["segment_ids"]
        sub_heads = example["sub_heads"] 
        sub_tails = example["sub_tails"] 
        obj_heads = example["obj_heads"] 
        
        obj_tails = example["obj_tails"] 
        sub_head = example["sub_head"]
        sub_tail = example["sub_tail"]
        
        text = example["text"]
        tokens = example["tokens"]
        
#         print(text)
#         print(tokens)
        
#         print(token_ids)
#         print(attention_mask)
#         print(sub_heads)
#         print(sub_tails)
        
#         input()
        
        example = [
            torch.tensor(token_ids).long().to(config.device),
            torch.tensor(attention_mask).long().to(config.device),
            torch.tensor(segment_ids).long().to(config.device),
            torch.tensor(sub_heads).float().to(config.device),
            torch.tensor(sub_tails).float().to(config.device),
            torch.tensor(obj_heads).float().to(config.device),
            torch.tensor(obj_tails).float().to(config.device),
            torch.tensor(sub_head).long().to(config.device),
            torch.tensor(sub_tail).long().to(config.device), 
            text,
            tokens
        ]
        
        return example
    
    def __len__(self):
        
        return self.data_len
    
class CustomBatch:
    
    def __init__(self, data):
        
        transposed_data = list(zip(*data))
        
        self.token_ids = torch.stack(transposed_data[0], 0)
        self.attention_mask = torch.stack(transposed_data[1], 0)
        self.segment_ids = torch.stack(transposed_data[2], 0)
        self.sub_heads = torch.stack(transposed_data[3], 0)
        self.sub_tails = torch.stack(transposed_data[4], 0)
        self.obj_heads = torch.stack(transposed_data[5], 0)
        self.obj_tails = torch.stack(transposed_data[6], 0)
        self.sub_head = torch.stack(transposed_data[7], 0)
        self.sub_tail = torch.stack(transposed_data[8], 0)
        self.text = transposed_data[9]
        self.tokens = transposed_data[10]
        
def collate_wrapper(batch):
    
    return CustomBatch(batch)  
        
        
import sys
sys.path.append("../")

from tqdm import tqdm
import json
import numpy as np
import torch

import config

def metric(lm_model, subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    
    
    if output_path:
        F = open(output_path, "w")
    orders = ["subject", "relation", "object"]
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(lm_model, subject_model, object_model, tokenizer, line["text"], id2rel))
        
#         print(Pred_triples)
        
        Gold_triples = set(line["triple_list"])
        
        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)
        
#         print(Pred_triples_eval)
        
#         print(Gold_triples_eval)
        
#         input()
        
        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)
        
        if output_path:
            
            result = json.dumps({"text": line["text"], 
                                 "triple_list_gold": [
                                     dict(zip(orders, triple)) for triple in Gold_triples
                                 ], 
                                 "triple_list_pred": [
                                     dict(zip(orders, triple)) for triple in Pred_triples
                                 ], 
                                 "new": [
                                     dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                                 ], 
                                 "lack":[
                                     dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                                 ]}, ensure_ascii=False, indent=4)
            F.write(result + "\n")
    if output_path:
        F.close()
            
    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)
        
    print(f"correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}")
    
    return precision, recall, f1_score


def extract_items(lm_model, subject_model, object_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):
    
    cls_token_segment_id = 0
    sequence_a_segment_id = 0
    
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(text_in) + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(token_ids)
    segment_ids = [cls_token_segment_id] + ([sequence_a_segment_id] * len(token_ids))[1:]
    
    if len(token_ids) > config.BERT_MAX_LEN:
        token_ids = token_ids[:config.BERT_MAX_LEN]
        segment_ids = segment_ids[:config.BERT_MAX_LEN]
        attention_mask = attention_mask[:config.BERT_MAX_LEN]
    
    token_ids = torch.tensor([token_ids]).long().to(config.device)
    attention_mask = torch.tensor([attention_mask]).long().to(config.device)
    segment_ids = torch.tensor([segment_ids]).long().to(config.device)
    
    bert_outputs = lm_model(token_ids, attention_mask, segment_ids)
    sub_heads_logits, sub_tails_logits = subject_model(bert_outputs)
    
    sub_heads, sub_tails = np.where(sub_heads_logits.cpu().numpy()[0] > h_bar)[0], np.where(sub_tails_logits.cpu().numpy()[0] > t_bar)[0]
    
    subjects = []
    for sub_head in sub_heads:
        
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            subject = tokens[sub_head: sub_tail]
            subjects.append((subject, sub_head, sub_tail))
            
    if subjects:
        triple_list = []
        token_ids = np.repeat(token_ids.cpu().numpy(), len(subjects), 0)
        segment_ids = np.repeat(segment_ids.cpu().numpy(), len(subjects), 0)
        attention_mask = np.repeat(attention_mask.cpu().numpy(), len(subjects), 0)
        
        token_ids = torch.tensor(token_ids).long().to(config.device)
        attention_mask = torch.tensor(attention_mask).long().to(config.device)
        segment_ids = torch.tensor(segment_ids).long().to(config.device)
        
        bert_outputs = lm_model(token_ids, attention_mask, segment_ids)
        sub_heads_logits, sub_tails_logits = subject_model(bert_outputs)
        
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
                
        obj_heads_logits, obj_tails_logits = object_model(bert_outputs, torch.tensor(sub_heads).long().to(config.device).view(-1, ), torch.tensor(sub_tails).long().to(config.device).view(-1, ))
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = "".join([i.lstrip("##") for i in sub])
            sub = "".join(sub.split("[unused1]"))
            obj_heads, obj_tails = np.where(obj_heads_logits.cpu()[i] > h_bar), np.where(obj_tails_logits.cpu()[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail]
                        obj = "".join([i.lstrip("##") for i in obj])
                        obj = "".join(obj.split("[unused1]"))
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []
    
def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold
        
                        
    
    
    
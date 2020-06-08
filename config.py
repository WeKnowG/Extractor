from transformers import BertConfig, DistilBertConfig, AlbertConfig
from utils.tokenizer import BERTTokenizer
from models import BERT
from transformers.modeling_bert import BertConfig
import torch

epoches = 200
batch_size = 32
lr = 1e-5

BERT_MAX_LEN = 200
RANDOM_SEED = 2020

max_text_len = 100

bert_feature_size = 1024


dataset_path = "./datasets/"
pretrained_model_path = "./pretrained_models/"
save_weights_path = "./saved_weights/"

MODEL_CLASSES = {
    'bert': (BertConfig, BERT, BERTTokenizer)
#     'distilbert': (DistilBertConfig, DistilBERT, DistilBertTokenizer),
#     'albert': (AlbertConfig, Albert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-large-cased'
#     'distilbert': 'distilbert-base-uncased',
#     'albert': 'albert-xxlarge-v1'
}

device = torch.device("cuda:0")

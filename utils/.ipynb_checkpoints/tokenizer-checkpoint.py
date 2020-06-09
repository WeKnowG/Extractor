import sys
sys.path.append("../")

import numpy as np

import unicodedata
import codecs
from tqdm import tqdm
import json
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
import config

class BERTTokenizer(BertTokenizer):
    
    def _tokenize(self, text):
        
#         split_tokens = []
#         if self.do_basic_tokenize:
#             for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

#                 # If the token is part of the never_split set
#                 if token in self.basic_tokenizer.never_split:
#                     split_tokens.append(token)
#                 else:
#                     split_tokens += self.wordpiece_tokenizer.tokenize(token)
#         else:
#             split_tokens = self.wordpiece_tokenizer.tokenize(text)
#             split_tokens.append("[unused1]")
        
        

        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        split_tokens = []
        for word in spaced.strip().split():
            split_tokens += self.wordpiece_tokenizer.tokenize(word)
            split_tokens.append('[unused1]')
            
        return split_tokens
    
    def _is_control(self, ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')


    
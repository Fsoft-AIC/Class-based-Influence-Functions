# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class BigClone(nn.Module):   
    def __init__(self, number_classes=2):
        super(BigClone, self).__init__()
        encoder, tokenizer, configer, block_size = config()
        self.encoder = encoder
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(configer)
        self.block_size = block_size
    
        
    def forward(self, input_ids=None): 
        input_ids=input_ids.view(-1,self.block_size)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        return logits

def load_BigClone_model(path, number_classes, device):
    model = BigClone(number_classes=number_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def build_BigClone_model(number_classes, device):
    model = BigClone(number_classes=number_classes)
    model = nn.DataParallel(model)
    model.to(device)
    return model


def config():
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    config = config_class.from_pretrained("microsoft/codebert-base",
                                          cache_dir=None)
    config.num_labels=2 
    tokenizer = tokenizer_class.from_pretrained("roberta-base",
                                                do_lower_case=False,
                                                cache_dir= None)
    encoder = model_class.from_pretrained("microsoft/codebert-base",
                                            from_tf=bool('.ckpt' in "microsoft/codebert-base"),
                                            config=config,
                                            cache_dir=None) 
    block_size = min(400, tokenizer.max_len_single_sentence)
    return encoder, tokenizer, config, block_size      
        
 
        



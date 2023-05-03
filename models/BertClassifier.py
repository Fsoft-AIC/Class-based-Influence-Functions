import torch.nn as nn
from transformers import BertConfig, BertModel
import torch

class BertClassifier(nn.Module):
    def __init__(self, output):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            config=config
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, output)
        nn.init.normal_(self.fc.weight, std=0.2)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, embedded = self.bert(input_ids, attention_mask, token_type_ids)
        x = self.dropout(embedded)
        out = self.fc(x)
        return out

    def _get_feature(self, input_ids, attention_mask, token_type_ids):
        _, embedded = self.bert(input_ids, attention_mask, token_type_ids)
        return embedded.squeeze(0) # (B, 512) --> (512)


def load_bert_model(path, number_classes, device):
    model = BertClassifier(output=number_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def build_bert_model(number_classes, device):
    model = BertClassifier(output=number_classes)
    model = nn.DataParallel(model)
    model.to(device)
    return model

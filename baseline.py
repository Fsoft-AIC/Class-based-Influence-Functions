import torch
from models.BertClassifier import load_bert_model, build_bert_model
from models.BigClone import load_BigClone_model, build_BigClone_model

from dataloaders.bigclone import bigClone_get_dataloader
from dataloaders.imdb import imdb_get_dataloader
from dataloaders.snli import snli_get_dataloader

class ModelBase():
    def __init__(self, model_type, number_classes, device='cuda'):
        self.model_type = model_type
        self.number_classes = number_classes
        self.device = device

    def build_model(self):
        if self.model_type == 'bert':
            self.model = build_bert_model(self.number_classes, self.device)
        elif self.model_type == 'BigCloneModel':
            self.model = build_BigClone_model(2, self.device)
            
        
    def load_model(self, path_pretrain):
        if self.model_type == 'bert':
            self.model = load_bert_model(path_pretrain, self.number_classes, self.device)
        elif self.model_type == 'BigCloneModel':
            self.model = load_BigClone_model(path_pretrain, 2, self.device)

    def inference(self, data):
        if self.model_type == 'bert':
            ids = data['ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            token_type_ids = data['token_type_ids'].to(self.device)
            labels = data['label'].to(self.device)
            predictions = self.model(ids, attention_mask, token_type_ids)

        elif self.model_type == 'BigCloneModel':
            sample, labels = data
            sample = sample.to(self.device)
            labels = labels.to(self.device)
            predictions = self.model(sample)

        return predictions, labels


class DataBase():
    def __init__(self, type_data):
        self.type_data = type_data

    def get_dataloader(self, df, batch_size, mode, num_workers=0):
        if self.type_data == 'bigclone':
            return bigClone_get_dataloader(
                df=df,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers
            )
        elif self.type_data == 'imdb':
            return imdb_get_dataloader(
                df=df,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers
            )
        elif self.type_data == 'snli':
            return snli_get_dataloader(
                df=df,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers
            )
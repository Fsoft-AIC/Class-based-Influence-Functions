import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from logging import raiseExceptions

class IMDBDataset(Dataset):
    def __init__(self, df, max_len=256):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df)

    def _get_input_data(self, row):
        text = row['review']
        
        tokens_text = self.tokenizer.tokenize(text)

        encode = self.tokenizer.convert_tokens_to_ids(tokens_text)
        if len(encode) > self.max_len - 2:
            encode = encode[:self.max_len - 2]
        # build ids
        ids = [self.tokenizer.cls_token_id] + encode + [self.tokenizer.sep_token_id]
        # token type ids
        token_type_ids = [0] + [0] * len(encode) + [0]
        # adding PAD token
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.tokenizer.pad_token_id] * pad_len
            token_type_ids += [self.tokenizer.pad_token_id] * pad_len
        # convert to tensor
        ids = torch.tensor(ids)
        token_type_ids = torch.tensor(token_type_ids)
        # Attention mask
        attention_mask = torch.where(ids!=0, torch.tensor(1), torch.tensor(0))

        return ids, attention_mask, token_type_ids
    
    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        ids, attention_mask, token_type_ids = self._get_input_data(row)
        label = int(row['label'])

        data['ids'] = ids
        data['attention_mask'] = attention_mask
        data['token_type_ids'] = token_type_ids
        data['label'] = label

        return data



def imdb_get_dataloader(df, batch_size, mode='train', num_workers=0):
    """ Get dataloader of pandas dataframe
    Args:
        df: Pandas Dataframe
        batch_size: batch size for dataloader
        mode (str, optional): ['train', 'test']. Defaults to 'train'.
        num_worker: number of worker for dataloader
    
    Return:
        dataloader: of DataFrame df
    """
    if mode == 'train':
        loader = DataLoader(
            IMDBDataset(df),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return loader

    elif mode == 'test':
        loader = DataLoader(
            IMDBDataset(df),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return loader

    else:
        return raiseExceptions("Mode does not support")


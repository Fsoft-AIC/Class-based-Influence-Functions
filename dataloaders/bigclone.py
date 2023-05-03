import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import json
import torch
import tqdm
from logging import raiseExceptions
import multiprocessing


def get_example(item):
    url1, url2, label, tokenizer, block_size, cache, url_to_code = item
    if url1 in cache:
        code1 = cache[url1].copy()
    else:
        try:
            code = ' '.join(url_to_code[url1].split())
        except:
            code = ""
        code1 = tokenizer.tokenize(code)
    if url2 in cache:
        code2 = cache[url2].copy()
    else:
        try:
            code = ' '.join(url_to_code[url2].split())
        except:
            code = ""
        code2 = tokenizer.tokenize(code)

    return convert_examples_to_features(code1, code2, label, url1, url2, tokenizer, block_size)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


def convert_examples_to_features(code1_tokens, code2_tokens, label, url1, url2, tokenizer, block_size):
    # source
    code1_tokens = code1_tokens[:block_size-2]
    code1_tokens = [tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:block_size-2]
    code2_tokens = [tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length

    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens, source_ids, label, url1, url2)

class BigCloneDataset(Dataset):
    def __init__(self, df, file_data_json, block_size = 400):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
                                                     do_lower_case=False,
                                                     cache_dir=None)
        self.examples = []
        url_to_code = {}
        with open(file_data_json) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']
        data = []
        cache = {}
        for index, line in df.iterrows():
            url1, url2, label = str(line['url1']), str(line['url2']), line['label']
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            if label == '1':
                label == 1
            data.append((url1, url2, label, tokenizer, block_size, cache, url_to_code))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            self.examples = pool.map(get_example, tqdm.tqdm(data, total=len(data)))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].label)

def bigClone_get_dataloader(df, batch_size, mode='train', num_workers=0):
    """ Get dataloader of pandas dataframe
    Args:
        df: Pandas Dataframe
        batch_size: batch size for dataloader
        mode (str, optional): ['train', 'test']. Defaults to 'train'.
        num_worker: number of worker for dataloader

    Return:
        dataloader: of DataFrame df
    """
    file_data_json = 'data/bigclone/dataset/data.jsonl'
    if mode == 'train':
        loader = DataLoader(
            BigCloneDataset(df, file_data_json),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader

    elif mode == 'test':
        loader = DataLoader(
            BigCloneDataset(df, file_data_json),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader

    else:
        return raiseExceptions("Mode does not support")

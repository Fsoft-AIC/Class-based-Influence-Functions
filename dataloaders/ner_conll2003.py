import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import numpy as np
from torch.utils.data import DataLoader
from logging import raiseExceptions
import os
import glob

def get_labels():
    return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def load_data_from_file(file_name, read_flipped_features):
    # you need create a file dataset as follow in a line: [required]Word [option]tag_POS_1 [option]tag_POS_2 [option]is_flip_label [option]tag_NER_origin [required]tag_NER
    f = open(file_name, "r")
    examples = []
    sentence = []
    label = []
    flipped = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            # if the end of sentence, we append this sentence to examples and reset all of lists
            if len(sentence) > 0:
                examples.append((sentence, label, flipped))
                sentence = []
                label = []
                flipped = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        # label.append(convert_label_to_single(splits[-1][:-1])) # convert label
        label.append(splits[-1][:-1])
        if read_flipped_features:
            flipped.append(splits[-2])
    
    if len(sentence) > 0:
        # convert example to InputFeature object
        examples.append((sentence, label, flipped))
        sentence = []
        label = []

    return examples


class CoNLL2003(Dataset):
    def __init__(self, file_name = None, read_flipped_features = False, max_len=128, path_gradient=None):
        super().__init__()
        self.read_flipped_features = read_flipped_features
        self.examples = load_data_from_file(file_name, read_flipped_features)
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.labels_to_ids = {label: i for i, label in enumerate(get_labels())}
        self.ids_to_labels = {i: label for i, label in enumerate(get_labels())}
        self.path_gradient = path_gradient

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.examples[index][0]
        word_labels = self.examples[index][1]
        flipped = self.examples[index][2]
        # print(sentence, word_labels)

        if self.path_gradient != None:
            g_i = torch.load(glob.glob(os.path.join(self.path_gradient + f'/*_{index}'))[0]) # g_i [list]: (128,)
            g_i = np.array([np.concatenate([np.reshape(w, -1) for w in token]) for token in g_i]) # np.narray (128, 6921)

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len
                                  )

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.labels_to_ids[label] for label in word_labels]

        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        # Word pieces that should be ignored have a label of -100 (which is the default ignore_index of PyTorch's CrossEntropyLoss).
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        if self.read_flipped_features:
            encoded_flipped = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                encoded_labels[idx] = labels[i]
                if self.read_flipped_features:
                    encoded_flipped[idx] = flipped[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels) #(bs, 128)
        if self.read_flipped_features:
            item['flipped'] = torch.as_tensor(encoded_flipped)
        if self.path_gradient != None:
            item['gradients'] = torch.as_tensor(g_i) # (bs, 128, 6921)
        # print(item)

        return item



def conll2003_get_dataloader(file_name, batch_size, mode='train', read_flipped_features = False, num_workers=0, path_gradient=None):
    if mode == 'train':
        loader = DataLoader(
            CoNLL2003(file_name, read_flipped_features, path_gradient=path_gradient),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader

    elif mode == 'test':
        loader = DataLoader(
            CoNLL2003(file_name, read_flipped_features, path_gradient=path_gradient),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader

    else:
        return raiseExceptions("Mode does not support")


if __name__ == '__main__':
    file_name = './data/conll2003/ner/noise_30sentence_30token.txt'
    path_gradient = 'checkpoints/conll2003/SEED0_NER_CoNLL2003_noise_data_30sentence_30token/noise_gradients'
    dataloader = conll2003_get_dataloader(file_name, batch_size=1, mode='test', read_flipped_features = True, path_gradient=path_gradient)
    for i, data in enumerate(dataloader):
        # print(data["flipped"])
        # print(data["labels"])15756MiB
        print(data)
        print(data['gradients'].size())
        print(data['labels'].size())
        print('-'*50)
        if  i == 3:
            break
    # data = CoNLL2003(file_name)
    # # data[1]
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # for i in [4]:
    #     for token, label in zip(tokenizer.convert_ids_to_tokens(data[i]["input_ids"]), data[i]["labels"]):
    #         print('{0:10}  {1}'.format(token, label))
    #     print('-'*20)
    # # pass

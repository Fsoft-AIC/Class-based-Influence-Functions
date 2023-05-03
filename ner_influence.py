import os
import torch
import tqdm
import numpy as np
from dataloaders.ner_conll2003 import conll2003_get_dataloader, get_labels
from transformers import BertTokenizerFast
from torch import linalg as LA

if __name__ == '__main__':
    noise_data_path = './data/conll2003/ner/noise_BItags_30sen_30tok.txt'
    noise_gradient_path = './checkpoints/conll2003/SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok/noise_gradients'


    test_data_path = './data/conll2003/ner/test.txt'
    test_gradient_path = './checkpoints/conll2003/SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok/stest_gradients'

    file_result = 'results/ner_IF_SEED4_BItags_noise_dataset.txt'

    batch_size = 128
    number_of_each_class = 10

    train_loader = conll2003_get_dataloader(
        file_name=noise_data_path,
        batch_size=batch_size,
        mode='test',
        num_workers=os.cpu_count(),
        read_flipped_features=True,
        path_gradient=noise_gradient_path
    )

    test_loader = conll2003_get_dataloader(
        file_name=test_data_path,
        batch_size=1,
        mode='test',
        num_workers=os.cpu_count(),
        read_flipped_features=False,
        path_gradient=test_gradient_path
    )

    # Build clean token
    print("Getting clean token...")
    lst_gradients_clean_data = []
    for ids_label in range(len(get_labels())):
        print("Find gradient of ids label", ids_label)
        number_samples = 0
        for i, sample in enumerate(test_loader):
            # print(sample)
            gradients = torch.as_tensor(sample["gradients"][0]) # (128, 6921)
            labels = sample["labels"][0] # (128,)
            for j, tag in enumerate(labels):
                if tag == ids_label and number_samples < number_of_each_class:
                    lst_gradients_clean_data.append(np.array(gradients[j]))
                    number_samples += 1
            if number_samples >= number_of_each_class:
                break

    # (num_of_each_class x number_class, 6921) = (num_token_clean, 6921)
    clean_gradients = torch.as_tensor(np.array(lst_gradients_clean_data))
        
    f = open(file_result, "w")
    # f.write("TOKEN,LABEL,FLIPPED,SCORE,SCORE_CLASS\n\n")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ids_to_labels = {i: label for i, label in enumerate(get_labels())}

    for item in tqdm.tqdm(train_loader):
        g_item = item["gradients"] # (bs, 128, 6921)
        scores = torch.tensordot(g_item, clean_gradients, dims = [[2], [1]]) #(bs, 128, num_token_clean)
        # norm_1 = LA.norm(g_item, dim=2) # (bs, 128)
        # norm_2 = LA.norm(clean_gradients, dim=1) # (num_token_clean)
        # scores = scores/torch.tensordot(norm_1.unsqueeze(-1), norm_2.unsqueeze(-1), dims=[[-1], [-1]])

        GD_normal = torch.sum(scores, dim=2)
        GD_class = torch.min(torch.as_tensor(np.array([np.array(torch.sum(scores[:,:,k:(k+1)*10], dim=2)) for k in range(len(get_labels()))])), dim=0).values
        tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in item['input_ids']]
        
        for token_sentence, mapping_sentence, label_sentence, flipped_sentence, score_sentence, score_class_sentence in zip(tokens, item["offset_mapping"].tolist(), item['labels'].tolist(), item['flipped'].tolist(), GD_normal.tolist(), GD_class.tolist()):
            #only predictions on first word pieces are important
            for token, mapping, label, flipped, score, score_class in zip(token_sentence, mapping_sentence, label_sentence, flipped_sentence, score_sentence, score_class_sentence):
                if mapping[0] == 0 and mapping[1] != 0:
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(token, ids_to_labels[label], flipped, score, score_class))
                else:
                    continue
            f.write("\n")

    f.close()


    # for item in tqdm.tqdm(train_loader):
    #     g_item = item["gradients"] # (bs, 128, 6921)
    #     scores = torch.tensordot(g_item, clean_gradients, dims = [[2], [1]]) #(bs, 128, num_token_clean)
    #     GD_normal = torch.sum(scores, dim=2)
    #     GD_class = torch.min(torch.as_tensor(np.array([np.array(torch.sum(scores[:,:,k:(k+1)*10], dim=2)) for k in range(len(get_labels()))])), dim=0).values
    #     tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in item['input_ids']]
        
    #     for token_sentence, mapping_sentence, label_sentence, score_sentence, score_class_sentence in zip(tokens, item["offset_mapping"].tolist(), item['labels'].tolist(), GD_normal.tolist(), GD_class.tolist()):
    #         #only predictions on first word pieces are important
    #         for token, mapping, label, score, score_class in zip(token_sentence, mapping_sentence, label_sentence, score_sentence, score_class_sentence):
    #             if mapping[0] == 0 and mapping[1] != 0:
    #                 f.write("{}\t{}\t{}\t{}\n".format(token, ids_to_labels[label], score, score_class))
    #             else:
    #                 continue
    #         f.write("\n")

    # f.close()
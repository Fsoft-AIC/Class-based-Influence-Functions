import os
import numpy as np
import torch
import tqdm
from numpy import save
from dataloaders.ner_conll2003 import get_labels, conll2003_get_dataloader
from models.BertSequence import load_bert_sequence_model
from influencer.GD import GD
from torch.autograd import grad

def build_gradient(model, params, dataloader, type_data = None):
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # if os.path.exists(os.path.join(dir_checkpoint, f"ck2_{type_data}_gradients", f'grad_{type_data}_{i}')):
        #     continue
        
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        labels = data['labels'].to(device, dtype = torch.long)
                
        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, reduction_loss = 'none')
        
        sentence_gradient = [grad(l, params, create_graph=False, retain_graph=True) for l in loss.view(-1)]
        lst_gradient_of_sentence = []

        for word_gradient in sentence_gradient:
            weight, bias = word_gradient
            weight = weight.cpu().detach().numpy()
            bias = bias.cpu().detach().numpy()
            word_gradient_numpy = [weight, bias]
            lst_gradient_of_sentence.append(word_gradient_numpy)
        
        # save gradient
        torch.save(lst_gradient_of_sentence, os.path.join(dir_checkpoint, f"ck2_{type_data}_gradients", f'grad_{type_data}_{i}'))


if __name__ == '__main__':
    # noise_BItags_30sen_30tok
    noise_file = os.path.join('data', 'conll2003', 'ner', 'noise_BItags_30sen_30tok.txt')
    dir_checkpoint = os.path.join('checkpoints', 'conll2003', 'SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok')
    device = 'cuda'
    num_labels = len(get_labels())
    type_data = "noise"
    SEED = 4
    name_checkpoint = 'epoch_2.pt' # best.pt

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not torch.cuda.is_available() and device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")


    # Load data
    noise_dataloader = conll2003_get_dataloader(
        file_name=noise_file,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    # Build model
    model = load_bert_sequence_model(os.path.join(dir_checkpoint, name_checkpoint),num_labels=num_labels,device=device)
    
    # Params to get gradients
    params = [p for p in model.parameters() if p.requires_grad][-2:]

    # get gradients
    build_gradient(
        model = model,
        params = params,
        dataloader = noise_dataloader,
        type_data = type_data
    )
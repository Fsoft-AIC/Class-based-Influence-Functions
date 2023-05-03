import os
import numpy as np
import torch
import tqdm
from models.BertSequence import load_bert_sequence_model
from dataloaders.ner_conll2003 import get_labels, conll2003_get_dataloader
import torch
from torch.autograd import grad

def ner_hvp(y, w, v):
    """ Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length.
    """
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length"))
    
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elementwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elementwise_products += torch.sum(grad_elem * v_elem)

    # second grad
    return_grads = grad(elementwise_products, w, create_graph=False, retain_graph=True)

    return return_grads

def ner_stest(params, zt, dataloader, num_sample = 1, damp=0.01, scale=25.0):
    ids = zt['input_ids'].to(device, dtype = torch.long)
    mask = zt['attention_mask'].to(device, dtype = torch.long)
    labels = zt['labels'].to(device, dtype = torch.long)
    loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, reduction_loss = 'none')
    # [list]: (128,)
    v_sentence = [grad(l, params, create_graph=False, retain_graph=True) for l in loss.view(-1)]
    
    stest_of_sentence = []
    for v in tqdm.tqdm(v_sentence):
        h_estimate = list(v).copy() # h_estimate ~ H(-1)v
        # Skip if all of gradients is zeros
        if torch.count_nonzero(v[0]).item() == 0 and torch.count_nonzero(v[1]).item() == 0:
            stest_of_sentence.append(h_estimate)
            continue
        
        for i, data in enumerate(dataloader):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)
            loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, reduction_loss = 'none')
            for l in loss:
                hv = ner_hvp(l, params, h_estimate)
                h_estimate = [_v + (1 - damp) * _he - _hv/scale for _v, _he, _hv in zip(v, h_estimate, hv)]
            
            if i == num_sample:
                break

        stest_of_sentence.append(h_estimate)
    
    return stest_of_sentence

if __name__ == '__main__':
    # configs
    dir_checkpoint = os.path.join('checkpoints', 'conll2003', 'SEED4_NER_CoNLL2003_noise_BItags_30sen_30tok')
    noise_file = os.path.join('data', 'conll2003', 'ner', 'noise_BItags_30sen_30tok.txt')
    test_file = os.path.join('data', 'conll2003', 'ner', 'test.txt')
    folder_save = os.path.join(dir_checkpoint, "stest_gradients")
    num_labels = len(get_labels())
    num_iteration=300
    scale = 25.0

    SEED = 4
    device = 'cuda'
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not torch.cuda.is_available() and device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)


    # Build model
    model = load_bert_sequence_model(os.path.join(dir_checkpoint, 'best.pt'),num_labels=num_labels,device=device)
    
    # Params to get gradients
    params = [p for p in model.parameters() if p.requires_grad][-2:]

    # dataloader
    noise_dataloader = conll2003_get_dataloader(
        file_name=noise_file,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    test_dataloader = conll2003_get_dataloader(
        file_name=test_file,
        batch_size=1,
        mode='test',
        num_workers=0
    )
    

    for i, zt in enumerate(test_dataloader):
        print("Sample: {}/{}".format(i+1, len(test_dataloader)))
        inverse_hvp = [torch.zeros_like(p, dtype=torch.float) for p in params]
        stest_of_sentence = ner_stest(params, zt, noise_dataloader)
        g_sentence = []
        for cur_estimate in stest_of_sentence:
            if torch.count_nonzero(cur_estimate[0]).item() == 0 and torch.count_nonzero(cur_estimate[1]).item() == 0:
                weight = cur_estimate[0].cpu().detach().numpy()
                bias = cur_estimate[1].cpu().detach().numpy()
                gt = [weight, bias]
                g_sentence.append(gt)
                continue

            for r in range(num_iteration):
                with torch.no_grad():
                    inverse_hvp = [old + (cur/scale) for old, cur in zip(inverse_hvp, cur_estimate)]
            with torch.no_grad():
                inverse_hvp = [j / num_iteration for j in inverse_hvp]
            weight, bias = inverse_hvp
            weight = weight.to('cpu')
            bias = bias.to('cpu')
            gt = [weight, bias]
            g_sentence.append(gt)
        
        # save gradient
        torch.save(g_sentence, os.path.join(folder_save, f'stest_{i}'))
import torch
import tqdm
from torch.autograd import grad

def exact_hessian(sample, inference_fn, params, loss_fn):
    preds, labels = inference_fn(sample)
    loss = loss_fn(preds, labels)
    loss_grad = grad(loss, params, retain_graph=True, create_graph=True)

    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1

    l = g_vector.size(0)
    hessian = torch.zeros(l,l)

    for idx in range(l):
        grad2rd = grad(g_vector[idx], params, retain_graph=True, create_graph=False)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous(
            ).view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2

    return hessian
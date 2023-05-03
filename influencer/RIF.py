import imp
import torch
import math
from tqdm import tqdm
from influencer.hessian import exact_hessian
from influencer.stest import build_stest

def RIF(test_loader, train_loader, test_gradients, train_gradients, inference_fn, loss_fn, params, use_exact_hessian = True, eps=0.1, device='cuda'):
    results = torch.zeros(len(test_gradients), len(train_gradients), dtype=float)
    for q, z in enumerate(tqdm(train_loader)):
        if use_exact_hessian:
            g = train_gradients[q]
            g = torch.cat([x.view(-1) for x in g])
            g = g.view(-1, 1).to(device)
            H = exact_hessian(z, inference_fn, params, loss_fn).to(device)
            H += torch.eye(H.shape[0]).to(device) * eps
            inverse_hessian = torch.inverse(H)
            sqrt_denominator = math.sqrt((g.T @ inverse_hessian @ g).item())
            for p, zt in enumerate(test_loader):
                gt = test_gradients[p]
                gt = torch.cat([x.view(-1) for x in gt])
                gt = gt.view(-1, 1).to(device)
                H_t = exact_hessian(zt, inference_fn, params, loss_fn).to(device)
                H_t += torch.eye(H_t.shape[0]).to(device) * eps
                inverse_hessian_t = torch.inverse(H_t)
                numerator = gt.T @ inverse_hessian_t @ gt
                numerator = numerator.item()
                results[p][q] = numerator/sqrt_denominator
        else:
            g = train_gradients[q]
            s = build_stest(z, inference_fn, loss_fn, params, train_loader, num_iteration=50)
            sqrt_denominator = math.sqrt(float(sum([torch.sum(k*j).data for k, j in zip(s, g)]).cpu().detach().numpy()))
            for p, zt in enumerate(test_loader):
                st = build_stest(zt, inference_fn, loss_fn, params, train_loader, num_iteration=50)
                numerator = sum([torch.sum(k * j).data for k, j in zip(st, g)])
                numerator = float(numerator.cpu().detach().numpy())
                results[p][q] = numerator/sqrt_denominator
    
    results = results.cpu().detach().numpy()
    return results
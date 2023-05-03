import torch
from tqdm import tqdm
from influencer.hessian import exact_hessian
from influencer.stest import build_stest

def IF(test_loader, train_loader, test_gradients, train_gradients, inference_fn, loss_fn, params, use_exact_hessian = True, eps=0.1, device='cuda'):
    results = torch.zeros(len(test_gradients), len(train_gradients), dtype=float)
    for p, zt in enumerate(tqdm(test_loader)):
        if use_exact_hessian:
            gt = test_gradients[p]
            gt = torch.cat([x.view(-1) for x in gt])
            gt = gt.view(-1, 1).to(device)
            H = exact_hessian(zt, inference_fn, params, loss_fn).to(device)
            H += torch.eye(H.shape[0]).to(device) * eps
            inverse_hessian = torch.inverse(H)
            for q, g in enumerate(train_gradients):
                g = torch.cat([x.view(-1) for x in g])
                g = g.view(-1, 1).to(device)
                influence = gt.T @ inverse_hessian @ g
                results[p][q] = influence.item()
        else:
            gt = build_stest(zt, inference_fn, loss_fn, params, train_loader)
            for q, g in enumerate(train_gradients):
                influence = sum([torch.sum(k * j).data for k, j in zip(gt, g)])
                influence = float(influence.cpu().detach().numpy())
                results[p][q] = influence
    results = results.cpu().detach().numpy()
    return results
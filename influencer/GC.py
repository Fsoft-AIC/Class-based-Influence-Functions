import torch
import tqdm
import torch.nn as nn

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

def GC(train_gradients, test_gradients):
    results = torch.zeros(len(test_gradients), len(train_gradients), dtype=float)
    
    for p, gt in enumerate(tqdm.tqdm(test_gradients)):
        gt = torch.cat([x.view(-1) for x in gt])
        for q, g in enumerate(train_gradients):
            g = torch.cat([x.view(-1) for x in g])
            influence = cos(gt, g).item()
            results[p][q] = influence
    results = results.cpu().detach().numpy()
    return results
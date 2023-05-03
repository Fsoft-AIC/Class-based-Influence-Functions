import torch
import tqdm

def GD(train_gradients, test_gradients):
    results = torch.zeros(len(test_gradients), len(train_gradients), dtype=float)
    for p, gt in enumerate(tqdm.tqdm(test_gradients)):
        for q, g in enumerate(train_gradients):
            influence = sum([torch.sum(k * j).data for k, j in zip(gt, g)])
            influence = float(influence.cpu().detach().numpy())
            results[p][q] = influence
    results = results.cpu().detach().numpy()
    return results
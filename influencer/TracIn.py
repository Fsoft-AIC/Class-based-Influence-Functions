import torch
import tqdm
from influencer.buildGradient import build_gradient

def TracIn(dir_checkpoint, model_base, train_loader, test_loader, loss_fn, start, end):
    results = torch.zeros(len(test_loader), len(train_loader), dtype=float) 
    for epoch in range(start, end+1):
        model_base.load_model(dir_checkpoint + '/epoch_' + str(epoch) + '.pt')
        params = [p for p in model_base.model.parameters() if p.requires_grad][-2:]
        
        train_gradients = build_gradient(
            inference_fn = model_base.inference,
            loss_fn=loss_fn,
            params=params,
            dataloader=train_loader
        )
        test_gradients = build_gradient(
            inference_fn = model_base.inference,
            loss_fn=loss_fn,
            params=params,
            dataloader=test_loader
        )
        for p, gt in enumerate(tqdm.tqdm(test_gradients)):
            for q, g in enumerate(train_gradients):
                influence = sum([torch.sum(k * j).data for k, j in zip(gt, g)])
                influence = float(influence.cpu().detach().numpy())
                results[p][q] += influence
        
        train_gradients, test_gradients = None, None
    results = results.cpu().detach().numpy()
    return results

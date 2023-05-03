import torch
import tqdm
from influencer.gradz import grad_z

def build_gradient(inference_fn, loss_fn, params, dataloader):
    gradients = []
    for data in tqdm.tqdm(dataloader):
        z_grad = grad_z(inference_fn, loss_fn, params, data, create_graph=False)
        # Send to cpu to Reduce memory
        weight, bias = z_grad
        weight = weight.to('cpu')
        bias = bias.to('cpu')
        gradients.append([weight, bias])
    return gradients
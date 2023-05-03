import torch
from influencer.gradz import grad_z
from influencer.hvp import hvp

def s_test(inference_fn, loss_fn, params, zt, dataloader, num_sample = 1, damp=0.01, scale=25.0):
    v = grad_z(inference_fn, loss_fn, params, zt, create_graph=False)
    h_estimate = v.copy() # h_estimate ~ H(-1)v
    for i, data in enumerate(dataloader):
        predictions, labels = inference_fn(data)
        loss = loss_fn(predictions, labels)
        hv = hvp(loss, params, h_estimate)
        h_estimate = [_v + (1 - damp) * _he - _hv/scale for _v, _he, _hv in zip(v, h_estimate, hv)]
        if i == num_sample:
            break

    return h_estimate

def build_stest(zt, inference_fn, loss_fn, params, train_loader, num_iteration=300, scale=25.0):
    inverse_hvp = [torch.zeros_like(p, dtype=torch.float) for p in params]
    cur_estimate = s_test(inference_fn, loss_fn, params, zt, train_loader)
    for r in range(num_iteration):
        with torch.no_grad():
            inverse_hvp = [old + (cur/scale) for old, cur in zip(inverse_hvp, cur_estimate)]
    with torch.no_grad():
        inverse_hvp = [j / num_iteration for j in inverse_hvp]
    weight, bias = inverse_hvp
    weight = weight.to('cpu')
    bias = bias.to('cpu')
    gt = [weight, bias]
    return gt
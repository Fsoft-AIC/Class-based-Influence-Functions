from torch.autograd import grad

def grad_z(inference_fn, loss_fn, params, data, create_graph=False):
    prediction, label = inference_fn(data)
    loss = loss_fn(prediction, label)
    return list(grad(loss, params, create_graph=create_graph))
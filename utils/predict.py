import torch
import tqdm


def predict(model, dataloader, func_inference):
    """ Predict and return a list of results on a data

    Args:
        model: Pytorch model
        dataloader: Dataloader of data
        func_inference (function): funtions that inference of model

    Returns:
        y_true: label of samples
        y_pred: prediction of model on dataloader
    """
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            predictions, labels = func_inference(data)
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred

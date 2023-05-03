import tqdm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def run_train(model, dataloader, optimizer, criterion, func_inference, mode='train'):
    """ Train function for model

    Args:
        model: model of ModelBase
        dataloader: dataloader of data
        optimizer: optimizer to optimization
        criterion: Cross entropy loss function
        func_inference (function): funtions that inference of model
        mode (str, optional): ['train', 'val']. Defaults to 'train'.

    Returns:
        epoch_loss: average loss of epoch
        epoch_acc: average accuracy of epoch
    """

    epoch_loss = 0.0
    preds, labs = [], []
    
    if mode == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    for data in tqdm.tqdm(dataloader):
        predictions, labels = func_inference(data)
        loss = criterion(predictions, labels)

        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        predictions = torch.max(torch.softmax(
            predictions, dim=1), dim=1).indices
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        preds.extend(predictions)
        labs.extend(labels)

    # print(classification_report(labs, preds))
    metrics = {"{}_acc".format(mode): accuracy_score(labs, preds),
               "{}_f1".format(mode): f1_score(labs, preds, average="weighted"),
               "{}_precision".format(mode):  precision_score(labs, preds, average="weighted"),
               "{}_recall".format(mode): recall_score(labs, preds, average="weighted"),
               "{}_loss".format(mode): epoch_loss/len(dataloader)}

    return metrics

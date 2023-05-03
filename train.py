import yaml
import os
import argparse
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from baseline import ModelBase, DataBase
from utils.run_train import run_train

def BinaryCrossEntropy(preds, labels):
    probs = torch.sigmoid(preds)
    labels = labels.float()
    loss = torch.log(probs[:,0]+1e-10)*labels+torch.log((1-probs)[:,0]+1e-10)*(1-labels)
    loss = -loss.mean()
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='config yaml path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', default='ClassTracing')
    opt = parser.parse_args()
    
    SEED = opt.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    print("Config training:")
    for key, value in cfg.items():
        print("{}: {}".format(key, value))

    if not os.path.isdir(cfg['dir_checkpoint']):
        print(f"Directory {cfg['dir_checkpoint']} does not exist")
        os.makedirs(cfg['dir_checkpoint'])
        print(f"Created {cfg['dir_checkpoint']}")

    if not torch.cuda.is_available() and cfg['device'] == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if cfg['device'] == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Build model
    model_base = ModelBase(cfg['model'],
                           cfg['number_classes'],
                           device=cfg['device'])
    model_base.build_model()

    # Load data
    df_train = pd.read_csv(cfg['df_train'])
    df_val = pd.read_csv(cfg['df_val'])

    data_base = DataBase(cfg['data'])

    train_loader = data_base.get_dataloader(
        df=df_train,
        batch_size=cfg['batch_size'],
        mode='train',
        num_workers=cfg['num_worker']
    )

    val_loader = data_base.get_dataloader(
        df=df_val,
        batch_size=cfg['batch_size'],
        mode='test',
        num_workers=cfg['num_worker']
    )

    # Loss function and optimizer
    if cfg['number_classes'] > 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = BinaryCrossEntropy
    
    optimizer = optim.AdamW(model_base.model.parameters(), lr=float(cfg['learning_rate']), betas=(0.9, 0.999))

    wandb.init(project=opt.project, name=cfg['name_project'])
    best_valid_acc = (-1.0) * float("Inf")
    best_epoch = 0

    for epoch in range(cfg['epoch']):
        print(f"Epoch: {epoch+1}/{cfg['epoch']}")
        train_metrics = run_train(
            model=model_base.model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            func_inference=model_base.inference,
            mode='train'
        )
        val_metrics = run_train(
            model=model_base.model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            func_inference=model_base.inference,
            mode='val'
        )

        print("Train metricc:", train_metrics)
        print("Val metrics", val_metrics)

        if cfg['save_each_epoch']:
            torch.save(model_base.model.state_dict(),
                       cfg['dir_checkpoint'] + '/epoch_{}.pt'.format(epoch))

        if best_valid_acc < val_metrics["val_acc"]:
            best_valid_acc = val_metrics["val_acc"]
            torch.save(model_base.model.state_dict(), cfg['dir_checkpoint'] + '/best.pt')
            f = open(cfg['dir_checkpoint'] + '/best_epoch.txt', "w")
            f.write(str(epoch))
            print(f"Model saved to ==> {cfg['dir_checkpoint'] + '/best.pt'} at epoch {epoch}")
        
        train_metrics.update(val_metrics)
        wandb.log(train_metrics)

    print(f'Finished training')

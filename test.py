import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from baseline import ModelBase, DataBase
from utils.run_train import run_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cifar10net') 
    parser.add_argument('--number-classes', type=int, default=10) 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-worker', default=0)
    parser.add_argument('--df', default='data/cifar10/processed/test.csv') 
    parser.add_argument('--type-data', default='cifar10') 
    parser.add_argument('--checkpoint', default='checkpoints/cifar10/SEED0_cifar10_cifar10net_train_knn_noise/best.pt')
    parser.add_argument('--seed', type=int, default=0) 
    
    opt = parser.parse_args()

    SEED = opt.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not torch.cuda.is_available() and opt.device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if opt.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Restone model
    model_base = ModelBase(opt.model,
                           opt.number_classes,
                           opt.device)
    model_base.load_model(opt.checkpoint)
    loss_fn = nn.CrossEntropyLoss()

    # Load data
    df = pd.read_csv(opt.df)
    print("Number of samples:", len(df))
    data_base = DataBase(opt.type_data)
    dataloader = data_base.get_dataloader(
        df=df,
        batch_size=opt.batch_size,
        mode='test',
        num_workers=opt.num_worker
    )

    metrics = run_train(
            model=model_base.model,
            dataloader=dataloader,
            optimizer=None,
            criterion=loss_fn,
            func_inference=model_base.inference,
            mode='test'
        )
    print("Metrics: {}".format(metrics))
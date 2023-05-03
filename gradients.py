import os
import yaml
import torch
import argparse
import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
from numpy import save
from baseline import ModelBase, DataBase
from influencer.IF import IF
from influencer.TracIn import TracIn
from influencer.buildGradient import build_gradient
from influencer.GD import GD
from influencer.GC import GC
from influencer.RIF import RIF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='config yaml path')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()

    SEED = opt.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    with open(opt.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    if not torch.cuda.is_available() and cfg['device'] == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if cfg['device'] == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # get name
    name_train = cfg['df_train'].split('/')[-1].split('.')[0]
    name_test = cfg['df_test'].split('/')[-1].split('.')[0]
    print("Name train: {}, Name test: {}".format(name_train, name_test))
    print("Directory: {}".format(cfg['dir_checkpoint']))

    # Load data
    df_train = pd.read_csv(cfg['df_train'])
    df_test = pd.read_csv(cfg['df_test'])

    data_base = DataBase(cfg['data'])

    train_loader = data_base.get_dataloader(
        df=df_train,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    test_loader = data_base.get_dataloader(
        df=df_test,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    # Config model
    model_base = ModelBase(
        model_type=cfg['model'],
        number_classes=cfg['number_classes'],
        device=cfg['device']
    )
    model_base.load_model(os.path.join(cfg['dir_checkpoint'],'best.pt'))
    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model_base.model.parameters() if p.requires_grad][-2:]
    
    # Build gradient
    if not os.path.exists(os.path.join(cfg['dir_checkpoint'],f'{name_train}.grad')):
        train_gradients = build_gradient(
            inference_fn = model_base.inference,
            loss_fn = loss_fn,
            params = params,
            dataloader = train_loader
        )
        torch.save(train_gradients, os.path.join(cfg['dir_checkpoint'],f'{name_train}.grad'))
    else:
        train_gradients = torch.load(os.path.join(cfg['dir_checkpoint'], f'{name_train}.grad'))
    
    if not os.path.exists(os.path.join(cfg['dir_checkpoint'],f'{name_test}.grad')):
        test_gradients = build_gradient(
            inference_fn = model_base.inference,
            loss_fn = loss_fn,
            params = params,
            dataloader = test_loader
        )
        torch.save(test_gradients, os.path.join(cfg['dir_checkpoint'], f'{name_test}.grad'))
    else:
        test_gradients = torch.load(os.path.join(cfg['dir_checkpoint'], f'{name_test}.grad'))
    
    # Run methods
    if 'IF' in cfg['methods']:
        print("Run Influence Function:")
        results = IF(
            test_loader=test_loader,
            train_loader=train_loader,
            test_gradients=test_gradients,
            train_gradients=train_gradients,
            inference_fn=model_base.inference,
            loss_fn=loss_fn,
            params=params,
            use_exact_hessian=cfg['use_exact_hessian']
        )
        if cfg['use_exact_hessian']:
            save(os.path.join(cfg['dir_checkpoint'],f'IF_exact_{name_train}_{name_test}'), results)
        else:
            save(os.path.join(cfg['dir_checkpoint'],f'IF_approximate_{name_train}_{name_test}'), results)

    if 'GD' in cfg['methods']:
        print("Run Grad-Dot:")
        results = GD(train_gradients, test_gradients)
        save(os.path.join(cfg['dir_checkpoint'],f'GD_{name_train}_{name_test}'), results)

    if 'GC' in cfg['methods']:
        print("Run Grad-Cos:")
        results = GC(train_gradients, test_gradients)
        save(os.path.join(cfg['dir_checkpoint'],f'GC_{name_train}_{name_test}'), results)

    if 'RIF' in cfg['methods']:
        print("Run RelatIF:")
        results = RIF(
            test_loader=test_loader,
            train_loader=train_loader,
            test_gradients=test_gradients,
            train_gradients=train_gradients,
            inference_fn=model_base.inference,
            loss_fn=loss_fn,
            params=params,
            use_exact_hessian=cfg['use_exact_hessian']
        )
        if cfg['use_exact_hessian']:
            save(os.path.join(cfg['dir_checkpoint'],f'RIF_exact_{name_train}_{name_test}'), results)
        else:
            save(os.path.join(cfg['dir_checkpoint'],f'RIF_approximate_{name_train}_{name_test}'), results)

    if 'TracIn' in cfg['methods']:
        start = 0
        f = open(os.path.join(cfg['dir_checkpoint'],'best_epoch.txt'))
        end = int(f.readline())
        print("Run Tracin from epoch {} to epoch {}".format(start, end))
        results = TracIn(
            dir_checkpoint=cfg['dir_checkpoint'],
            model_base=model_base,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            start=start,
            end=end
        )
        save(os.path.join(cfg['dir_checkpoint'],f'TracIn_{name_train}_{name_test}'), results)
        

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import load
import torch
import glob
import os
import random
import tqdm
import torch.nn as nn
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    data = 'bigclone'
    model = 'codebert'
    percent = '20'
    train_gradient = torch.load(os.path.join('checkpoints', f'{data}', f'{data}_{model}_train_noise_{percent}%', f'noise{percent}%.grad'))
    # train_features = load(os.path.join('checkpoints', f'{data}', f'{data}_{model}_train_random_noise', 'train_features.npy'))
    pd_data = pd.read_csv(f'data/{data}/processed/noise/noise{percent}%.csv')
    
    min_class = min(set(pd_data['label']))
    max_class = max(set(pd_data['label']))
    
    # gradient Dot all
    samples = random.sample(train_gradient, 1000)
    results = torch.zeros(len(samples), len(samples), dtype=float)
        
    for p, gt in enumerate(tqdm.tqdm(samples)):
        for q, g in enumerate(samples):
            influence = sum([torch.sum(k * j).data for k, j in zip(gt, g)])
            influence = float(influence.cpu().detach().numpy())
            results[p][q] = influence
    results = results.cpu().detach().numpy()

    values = []
    for i in range(1000-1):
        for j in range(i+1, 1000):
            values.append(results[i][j])

    fig, ax = plt.subplots(1, 1)
    ax.hist(values, bins=100)
    plt.savefig(f'figures/{data}/dot/{data}_all_grad_dot.pdf')

    #  Gradient Cos all
    results = torch.zeros(len(samples), len(samples), dtype=float)
        
    for p, gt in enumerate(tqdm.tqdm(samples)):
        gt = torch.cat([x.view(-1) for x in gt])
        for q, g in enumerate(samples):
            g = torch.cat([x.view(-1) for x in g])
            influence = cos(gt, g).item()
            results[p][q] = influence
    results = results.cpu().detach().numpy()

    values = []
    for i in range(1000-1):
        for j in range(i+1, 1000):
            values.append(results[i][j])

    fig, ax = plt.subplots(1, 1)
    ax.hist(values, bins=100)
    plt.savefig(f'figures/{data}/cos/{data}_all_grad_cos.pdf')

    # gradient dot of a class
    for c in range(min_class, max_class + 1):
        data_class = pd_data[pd_data['label']==c][:1000] # lay 1000 samples moi class
        n = len(data_class.index)
        results =  np.zeros((n,n))
        for p, gt in enumerate(tqdm.tqdm(range(n))):
            gt = train_gradient[data_class.index[p]]
            for q, g in enumerate(range(n)):
                g = train_gradient[data_class.index[q]]
                influence = sum([torch.sum(k * j).data for k, j in zip(gt, g)])
                influence = float(influence.cpu().detach().numpy())
                results[p][q] = influence 
        values = []
        for i in range(n-1):
            for j in range(i+1, n):
                values.append(results[i][j])
        fig, ax = plt.subplots(1, 1)
        ax.hist(values, bins=100)
        plt.savefig(f'figures/{data}/dot/{data}_grad_dot_class_{c}.pdf')
    
    # Ve cos tung class theo gradient
    for c in range(min_class, max_class + 1):
        data_class = pd_data[pd_data['label']==c][:1000] # lay 1000 samples
        n = len(data_class.index)
        results =  np.zeros((n,n))
        for p, gt in enumerate(tqdm.tqdm(range(n))):
            gt = train_gradient[data_class.index[p]]
            gt = torch.cat([x.view(-1) for x in gt])
            for q, g in enumerate(range(n)):
                g = train_gradient[data_class.index[q]]
                g = torch.cat([x.view(-1) for x in g])
                influence = cos(gt, g).item()
                results[p][q] = influence 
        values = []
        for i in range(n-1):
            for j in range(i+1, n):
                values.append(results[i][j])
        fig, ax = plt.subplots(1, 1)
        ax.hist(values, bins=100)
        plt.savefig(f'figures/{data}/cos/{data}_grad_cos_class_{c}.pdf')
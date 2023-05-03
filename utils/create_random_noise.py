import pandas as pd
import numpy as np
import argparse
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default=0.2)
    parser.add_argument('--data', required=True, help="Name of dataset")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='random_noise')
    opt = parser.parse_args()

    np.random.seed(opt.seed)
    random.seed(opt.seed)

    df = pd.read_csv(f"data/{opt.data}/processed/train.csv")
    start_label = min(set(df['label']))
    end_label = max(set(df['label']))
    df['isFlipped'] = [0] * len(df)
    df['originLabel'] = df['label']
    number_change = int(len(df)*opt.percent)
    indexes_change = []

    while len(indexes_change) < number_change:
        ind = random.choice(np.arange(0, len(df)))
        if ind not in indexes_change:
            indexes_change.append(ind)

    label_origin = df.loc[indexes_change, 'originLabel'].tolist()

    label_new = np.random.randint(start_label, end_label + 1, size=int((len(df)*opt.percent)),)

    for i in range(len(label_origin)):
        if label_origin[i] == label_new[i]:
            l = label_origin[i]
            while l == label_origin[i]:
                l = np.random.randint(start_label, end_label+1)
            label_new[i] = l

    df.loc[indexes_change, 'label'] = label_new
    df.loc[indexes_change, 'isFlipped'] = 1

    print("Flipped: {}/{} samples".format(len(indexes_change), len(df)))
    df.to_csv(f"data/{opt.data}/processed/{opt.filename}.csv", index=False)

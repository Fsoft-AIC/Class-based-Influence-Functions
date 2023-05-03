import argparse
import pandas as pd
import numpy as np
from numpy import load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path of result file')
    # parser.add_argument('--data', type=str, required=True, help="name of dataset")
    parser.add_argument('--step', type=int, required=True, help="step of class in test data")
    parser.add_argument('--train', required=True, help='path of train data')
    parser.add_argument('--test', required=True, help='path of test data')
    opt = parser.parse_args()
    result = load(opt.path)
    result = pd.DataFrame(result)
    # print(result)
    # train_path = f"data/{opt.data}/processed/random_noise.csv"
    # test_path = f"data/{opt.data}/processed/subclass.csv"
    df_train = pd.read_csv(opt.train)
    df_test = pd.read_csv(opt.test)
    # df_train = pd.read_csv(train_path)
    # df_test = pd.read_csv(test_path)
    start_class = min(set(df_train['label']))
    end_class = max(set(df_train['label']))
    num_class = len(set(df_train['label']))

    results1 = []
    top = [0.05, 0.1, 0.15, 0.2]
    # top = np.arange(0.05, 0.21, 0.01)
    # top = [0.05]
    top = [int(len(df_train) * i) for i in top]
    n = len(df_train)
    # print("Calculate by all samples")
    for t in top:
        first = (df_train.iloc[np.argsort(result.sum().values)]['isFlipped'][:t] == 1).sum()
        last = (df_train.iloc[np.argsort(result.sum().values)]['isFlipped'][-t:] == 1).sum()
        results1.append(first/t*100)
        # print(first/t*100, last/t*100)

    print("OLD:")
    print(results1)
    # print("Calculate by class")
    # create score for each sample
    df_class = pd.DataFrame()
    i = 0
    while i< num_class:
        class_scores = result.iloc[i*opt.step:(i+1)*opt.step + 1].sum()
        # pd.concat([df_class, class_scores], axis=1)
        df_class[i] = class_scores.values
        i+=1
    # print(df_class)
    scores = df_class.min(axis=1).values
    # print(scores)
    results2 = []
    for t in top:
        first = (df_train.iloc[np.argsort(scores)]['isFlipped'][:t] == 1).sum()
        last = (df_train.iloc[np.argsort(scores)]['isFlipped'][-t:] == 1).sum()
        results2.append(first/t*100)
        # print(first/t*100, last/t*10)

    # print(results)
    # for item in results1:
    #     print("{:.2f}".format(item), end='\t')
    # print()
    # for item in results2:
    #     print("{:.2f}".format(item), end='\t')
    # print()
    print("NEW:")
    print(results2)
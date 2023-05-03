import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='name of dataset')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='subclass')
    opt = parser.parse_args()

    data = pd.read_csv(f"data/{opt.data}/processed/val.csv")
    min_label = min(set(data['label']))
    max_label = max(set(data['label']))
    result = pd.DataFrame()
    for label in range(min_label, max_label + 1):
        label_data = data[data['label'] == label]
        result = result.append(label_data.sample(n=opt.samples, random_state=opt.seed))
    result.reset_index(inplace=True, drop=True)
    print(result)
    result.to_csv(f"data/{opt.data}/processed/subclass/{opt.filename}.csv", index=False)

import numpy as np
import pandas as pd
import os

split_ratio = 0.8


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('./文本分类练习.csv')
    data['length'] = data['review'].apply(lambda x: len(x))
    max_length = data['length'].max()
    data.drop(columns=['length'], inplace=True)
    data['label'] = data['label'].apply(lambda x: '好评' if x == 1 else '差评')
    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    # Split data
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx]
    valid_data = data.iloc[split_idx:]
    # Save data to json 
    train_data.to_json('./data/train_data.json', orient='records', lines=True, force_ascii=False)
    valid_data.to_json('./data/valid_data.json', orient='records', lines=True, force_ascii=False)
    print('Max length:', max_length)
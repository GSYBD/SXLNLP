import pandas as pd
from sklearn.model_selection import train_test_split
import json

def split_data(input_path, train_tag_review, valid_tag_review, valid_size=0.2, random_state=42):
    # 读取CSV数据
    df = pd.read_csv(input_path)

    # 确保数据有两列：第一列是label，第二列是review
    if df.shape[1] != 2:
        raise ValueError("输入的CSV文件应当有两列：第一列是label，第二列是review")

    # 分割数据集
    train_df, val_df = train_test_split(df, test_size=valid_size, random_state=random_state, stratify=df.iloc[:, 0])

    # 将数据集保存为JSON格式
    train_data = train_df.to_dict(orient='records')
    val_data = val_df.to_dict(orient='records')

    with open(train_tag_review, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(valid_tag_review, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    print(f"训练集已保存至 {train_tag_review}")
    print(f"验证集已保存至 {valid_tag_review}")


if __name__ == "__main__":
    input_path = '../Data/文本分类练习.csv'
    train_tag_review = '../Data/train_tag_review.json'
    valid_tag_review = '../Data/valid_tag_review.json'
    split_data(input_path, train_tag_review, valid_tag_review)

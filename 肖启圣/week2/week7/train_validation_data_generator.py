import pandas as pd
df = pd.read_csv(r'C:\Users\15007\Desktop\Python Learn\Homework\Week7\文本分类练习.csv')
shuffled_df = df.sample(frac=1, random_state=42)
# shuffled_df.to_csv(r'C:\Users\15007\Desktop\Python Learn\Homework\Week7\evaluate_data.csv', index=False)

train_df = df.iloc[:8400]
train_df.to_csv(r'C:\Users\15007\Desktop\Python Learn\Homework\Week7\train_data.csv', index=False)

evaluate_df = df.iloc[8400:]
evaluate_df.to_csv(r'C:\Users\15007\Desktop\Python Learn\Homework\Week7\evaluate_data.csv', index=False)
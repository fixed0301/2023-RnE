import pandas as pd


df = pd.read_csv('../lm_csv_test1/df1_test1.csv')
tmp = df.values.tolist()

print(tmp)
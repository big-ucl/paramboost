import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = pd.concat([df_train, df_test])

df.to_csv('har.csv', index=False)
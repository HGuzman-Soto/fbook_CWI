import pandas as pd


df = pd.read_csv('data.csv')

df = df[['id', 'text', 'starting_index', 'ending_index', 'content_word']]
print(df.head())

df.to_csv('test_data.tsv', index=False, sep='\t')

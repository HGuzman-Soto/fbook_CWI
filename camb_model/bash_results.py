import pandas as pd
import glob

file = glob.glob("results/*.csv")[-1]
df = pd.read_csv(file)
df = df[['sentence', 'word', 'output',]]
df.to_csv(file)
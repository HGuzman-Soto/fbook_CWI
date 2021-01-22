import os
import time
import glob
import shutil
import multiprocessing
import pandas as pd
from dask import dataframe as dd
from dask.multiprocessing import get
from preprocess import clean
from preprocess import extract_content_words
from preprocess import isValuableComment
from os import path
from dask.distributed import Client

from multiprocessing import cpu_count
nCores = cpu_count()

start_time = time.time()
# df = dd.read_csv('temp_data.csv')
df = pd.read_csv('temp_data.csv')

# df = df[1000:1500]

"""todo
1) rename this file
2) Use spacy.pipe() for all these processes/maybe the preprocess.py file to make things faster


Steps
1) We clean our social media data --> clean_comments
2) We apply a small heuristic to remove rows with less than 10 characters
3) As an intermediate step, we generate a list of all content words for each text
4) Then, using this list, we attach each word as a seperate row --> content_word
5) Finally, we attach the indexes, which are represented as a tuple --> index

"""

df['clean_text'] = df.text.apply(lambda x: clean(x))

df = df[df.clean_text.apply(lambda x: isValuableComment(x)) == True]
meta = ('content_word', 'object')



# convert to dask dataframe and call extract_content_words()
df = dd.from_pandas(df, npartitions=12)
df['content_word'] = df.map_partitions(lambda df: df.clean_text.apply(
    lambda x: extract_content_words(x)), meta=meta).compute(num_workers=nCores)

df = df.compute()  # dask to pandas again

# reference without dask
df['content_word'] = df.clean_text.apply(
    lambda x: extract_content_words(x))

# remove rows w/ no content words
df = df[df['content_word'].astype(bool)]

print("\n")
print("Expanding content word lists \n")
df = df.explode('content_word')

print("Attaching indexes to each content words \n")
df[['starting_index', 'ending_index']] = df.apply(lambda x: (x['content_word'][1][0], x['content_word'][1][1]), axis=1, result_type='expand')

# reformatting content words from set members back to single tokens
df['content_word'] = df['content_word'].apply(lambda x: x[0])

# df.to_csv('data.csv', index=False)
print("Finished")
print("--- %s seconds ---" % (time.time() - start_time))

# test = df.compute()
print(df)
if path.exists('data.csv'):
    df.to_csv('data.csv', mode='a', header=False, index=False)
    os.remove('temp_data.csv')

else:
    df.to_csv('data.csv', index=False)
    os.remove('temp_data.csv')

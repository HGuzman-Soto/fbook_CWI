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
df = pd.read_csv('temp_data.csv')

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

df['clean_sentence'] = df.sentence.apply(lambda x: clean(x))

df = df[df.clean_sentence.apply(lambda x: isValuableComment(x)) == True]
meta = ('word', 'object')


# convert to dask dataframe and call extract_content_words()
# df = dd.from_pandas(df, npartitions=12)
# df['word'] = df.map_partitions(lambda df: df.clean_sentence.apply(
#     lambda x: extract_content_words(x)), meta=meta).compute(num_workers=nCores)

# df = df.compute()  # dask to pandas again

# reference without dask
df['word'] = df.clean_sentence.apply(
    lambda x: extract_content_words(x))

# remove rows w/ no content words
df = df[df['word'].astype(bool)]

print("\n")
print("Expanding content word lists \n")
df = df.explode('word')

print("Attaching indexes to each content words \n")
df[['start_index', 'end_index']] = df.apply(lambda x: (
    x['word'][1][0], x['word'][1][1]), axis=1, result_type='expand')

# reformatting content words from set members back to single tokens
df['word'] = df['word'].apply(lambda x: x[0])

# df.to_csv('data.csv', index=False)
print("Finished")
print("--- %s seconds ---" % (time.time() - start_time))

df = df.reindex(columns=['ID', 'sentence', 'clean_sentence',
                         'start_index', 'end_index', 'word'])


# test = df.compute()

df.to_csv('data.csv', index=False)

data_num = len(os.listdir("data_files/data"))
temp_data_num = len(os.listdir("data_files/temp_data"))

new_data_name = "data_" + str(data_num) + ".csv"
new_temp_name = "temp_data_" + str(temp_data_num) + ".csv"

os.rename('data.csv', new_data_name)
os.rename('temp_data.csv',  new_temp_name)

shutil.move(new_data_name, "data_files/data/")
shutil.move(new_temp_name, "data_files/temp_data/")

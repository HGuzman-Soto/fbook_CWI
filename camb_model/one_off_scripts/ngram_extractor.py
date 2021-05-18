import nltk
import string
import pandas as pd
from nltk import tokenize
from nltk.util import ngrams
from collections import Counter
import numbers

from nltk import ngrams
from pandas.io.sql import table_exists



# the real code goes here
#####################################

lgrams = []
n = 2

wiki_df = pd.read_csv("C:/Users/rasmu/CWI/fbook_CWI/camb_model/corpus/spanish/wikipedia-esp.csv")
# wiki_df = wiki_df[:1000]
tcount = 0
germ_chars = 'abcdefghijklmnopqrstuvwxyzáéíóúüñ'

for index, row in wiki_df.iterrows():
    if isinstance(row['word'],numbers.Number):
        continue
    bigrams = ngrams(row['word'],n)
    temp_grams = []

    try:
        [temp_grams.append(str(gram[0] + gram[1])) for gram in bigrams if (gram[0] in germ_chars and gram[1] in germ_chars)]
        temp_grams * row['frequency']
        lgrams += temp_grams
        tcount += len(temp_grams)
    except:
        continue

#####################################
# stick the counts into a dataframe
print("FINSIHED")
c = Counter(lgrams)
df = pd.DataFrame.from_dict(c, orient='index').reset_index()
df = df.rename(columns={'index':'bigram', 0:'frequency'})

l_gramtypes = len(c)
df['probability'] = df['frequency'].apply(lambda x: x / tcount)
df = df.sort_values('frequency', ascending=False)

print(df)
df.to_csv("out.csv", index=False)
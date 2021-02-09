import pandas as pd
import csv
# Get dataset from https://nlp.cs.nyu.edu/wikipedia-data/


def main():
    df = pd.read_csv('wp_1gram.txt', delim_whitespace=True,
                     quoting=csv.QUOTE_NONE)
    nan_value = float("NaN")

    # casing
    df['word'] = df.word.str.lower()

    # remove punctuaction
    df['word'] = df.word.str.replace(
        '[^\w\s]', '')

    # remove numbers
    df['word'] = df.word.str.replace('\d+', '')

    # remove whitespace
    df['word'] = df.word.str.strip()
    df = df.dropna(subset=['word'])
    df['word'] = df.word[df.word.map(len) > 0]

    df['frequency'] = df.frequency.astype(int)
    df = df.dropna(subset=['word'])


    df = df[['word', 'frequency']]
    print(len(df))
    df.to_csv('wikipedia_corpus.csv', index=False)


main()

import pandas as pd


def main():
    df_combined = get_data()
    df_frequency = get_frequency(df_combined)
    convert_to_csv(df_frequency)


def get_data():
    # get dataset from https://sites.google.com/site/naistlang8corpora/home/readme-en

    fields = ['num_corrections', 'serial_num',
              'url', 'sentence_num', 'learner_eng', 'correction']
    df_test = pd.read_table("lang-8-en-1.0/entries.test",
                            usecols=fields)

    df_train = pd.read_table("lang-8-en-1.0/entries.train",
                             usecols=fields)

    df_train = df_train['learner_eng']
    df_test = df_test['learner_eng']

    merged_df = pd.concat([df_test, df_train], axis=0)
    merged_df = merged_df.astype(str)
    return merged_df


def get_frequency(df):
    # casing
    df = df.str.lower()

    # # remove stop words
    # df_train['learner_eng'] = df_train['learner_eng'].apply(lambda x: ' '.join(
    #     [word for word in x.split() if word not in (stop)]))

    # remove punctuaction
    df = df.str.replace(
        '[^\w\s]', '')

    # remove numbers
    df = df.str.replace('\d+', '')

    # remove whitespace
    df = df.str.strip()

    series_top = pd.Series(
        ' '.join(df).split()).value_counts()
    df_top = series_top.to_frame(name="frequency")
    df_top['word'] = df_top.index
    df_top = df_top[['word', 'frequency']]
    print(df_top.head)
    return df_top


def convert_to_csv(df):
    df.to_csv('corpus/learner_corpus.csv', index=False)


if __name__ == "__main__":
    main()

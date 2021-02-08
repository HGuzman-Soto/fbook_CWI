import pandas as pd


def main():
    df = get_data()
    convert_to_csv(df)


def get_data():
    # get dataset from http://www.kilgarriff.co.uk/bnc-readme.html#raw

    fields = ['id', 'word', 'pos', 'frequency']
    df = pd.read_table("one_off_scripts/bnc_data",
                       usecols=fields, sep=" ")

    df = df[['word', 'pos', 'frequency']]

    df['word'] = df['word'].astype(str)
    df['pos'] = df['pos'].astype(str)
    df['frequency'] = df['frequency'].astype(int)
    return df


def convert_to_csv(df):
    df.to_csv('corpus/bnc_corpus.csv', index=False)


if __name__ == "__main__":
    main()

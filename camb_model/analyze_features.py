import seaborn as sns
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import argparse
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer


def main(df):

    if(args.option == "chi"):
        chi_square(df)
    if(args.option == "heat"):
        heat_map(df)
    if(args.option == "matrix"):
        matrix_plot(df)

##########################################################################################################


def heat_map(df):
    # independent columns
    X = df.iloc[:, 6:]

    y = df['complex_binary']
    # get correlations of each features in dataset
    corrmat = X.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    sns.heatmap(df[top_corr_features].corr(),
                linewidths=.5, annot=True, fmt='.1f', cmap="RdYlGn")

    plt.show()

##########################################################################################################


def chi_square(df):
    X = df.iloc[:, 9:]
    y = df['complex_binary']

    vectorizer = CountVectorizer()
    words_test = vectorizer.fit_transform(X['pos'])
    X['pos'] = pd.DataFrame(
        words_test.toarray(), columns=vectorizer.get_feature_names())

    vectorizer = CountVectorizer()
    words_test = vectorizer.fit_transform(X['lemma'])
    X['lemma'] = pd.DataFrame(
        words_test.toarray(), columns=vectorizer.get_feature_names())

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(40, 'Score'))  # print 10 best features

    sns.barplot(x="Specs", y="Score", data=featureScores)
    plt.show()

##########################################################################################################


"""
Make like 5-7 matrix plot
"""


def matrix_plot(df):
    outputs = df['output']
    print(type(args.matrix_2))
    subset_df = df[[args.matrix_1, args.matrix_2,
                    args.matrix_3, args.matrix_4, args.matrix_5]]
    subset_df['output'] = outputs
    sns.set_theme(style="ticks")

    # df = sns.load_dataset("penguins")
    sns.pairplot(subset_df, hue="output")
    plt.show()
##########################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Plot data')
    parser.add_argument('data', type=str, default=None)
    parser.add_argument('option', type=str, default=None)
    parser.add_argument('-matrix_1', "-1",  type=str, default=None)
    parser.add_argument('-matrix_2', "-2",  type=str, default=None)
    parser.add_argument('-matrix_3', "-3", type=str, default=None)
    parser.add_argument('-matrix_4', "-4", type=str, default=None)
    parser.add_argument('-matrix_5', "-5",  type=str, default=None)

    args = parser.parse_args()
    return args
##########################################################################################################


if __name__ == "__main__":
    args = parse_args()

    data_name = args.data

    df = pd.read_csv("results/" + data_name + ".csv")
    main(df)

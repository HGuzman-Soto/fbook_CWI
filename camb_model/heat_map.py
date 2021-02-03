import seaborn as sns
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer


name = "wikipedia_test_all_data_ada_results"
data = pd.read_csv("results/" + name + ".csv")


def heat_map():
    # independent columns
    X = data.iloc[:, 9:31]
    print(X.describe())

    f, ax = plt.subplots(figsize=(18, 18))

    y = data['complex_binary']
    # get correlations of each features in dataset
    corrmat = X.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(data[top_corr_features].corr(),
                    linewidths=.5, annot=True, fmt='.1f', cmap="RdYlGn")

    plt.show()


def chi_square():
    X = data.iloc[:, 9:31]
    print(X.describe())
    y = data['complex_binary']

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
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features


def matrix_plot(data):
    outputs = data['output']
    data = data.iloc[:, 7:15]
    print(data.describe)
    sns.set_theme(style="ticks")

    # df = sns.load_dataset("penguins")
    sns.pairplot(data, hue="output")
    plt.show()


chi_square()


heat_map()
matrix_plot(data)

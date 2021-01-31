from os import pipe
from sklearn.naive_bayes import GaussianNB
import scipy.stats as stats
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import string
import numpy as np
import argparse
import pandas as pd
import sys
import pickle
import wandb
from wandb.keras import WandbCallback
from train_model import TextSelector, NumberSelector


def main():
    pass

##########################################################################################################


def apply_algorithm(name, array):
    loaded_model = pickle.load(
        open("models/" + args.model_name + ".sav", 'rb'))

    model_stats = pd.DataFrame(
        columns=['Data', 'Classifier', 'Precision', 'Recall', 'F-Score'])

    i = 0
    for x in array:
        print("results for", name, ":\n")
        data = x
        targets = data['complex_binary'].values
        predictions = loaded_model.predict(data)

        df = pd.DataFrame(data=data)
        df = df.drop(columns=['parse', 'count', 'split', 'original word',
                              'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])
        df['output'] = predictions
        df.to_csv("results/" + name + "_results.csv", index=False)

        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        F_Score = f1_score(targets, predictions, average='macro')

        model_stats.loc[len(model_stats)] = [i, (str(loaded_model))[
            :100], precision, recall, F_Score]

        model_stats.to_csv("results/" + name + "_metrics.csv", index=False)
        print("Accuracy", accuracy)
        print("Precision:", model_stats.Precision)
        print("Recall:", model_stats.Recall)
        print("F-Score:", model_stats['F-Score'], "\n")

##########################################################################################################


"""
To-Do -

Split apply_algorithm into three functions - Since the pipeline feature should make this fast

1) score algorithm - just handles getting all the metrics

2) Get results (append features + outputs)

3) ROC + other visualizations
"""


def score_model(name, array):
    model_stats = pd.DataFrame(
        columns=['Data', 'Classifier', 'Precision', 'Recall', 'F-Score'])
    i = 0
    for x in array:
        print("results for", name[i], ":\n")
        data = x
        targets = data['complex_binary'].values
        predictions = pipeline.predict(data)

        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        F_Score = f1_score(targets, predictions, average='macro')

        model_stats.loc[len(model_stats)] = [i, (str(model))[
            :100], precision, recall, F_Score]
        print("Accuracy", accuracy)
        print("Precision:", model_stats.Precision)
        print("Recall:", model_stats.Recall)
        print("F-Score:", model_stats['F-Score'], "\n")

##########################################################################################################


def get_results(name, array):
    i = 0
    for x in array:
        data = x
        targets = data['complex_binary'].values
        predictions = pipeline.predict(data)
        df = pd.DataFrame(data=data)
        df = df.drop(columns=['parse', 'count', 'split', 'original word'])
        df['output'] = predictions
        predict_df = df['output']
        df.drop(labels=['output'], axis=1, inplace=True)
        df.insert(7, 'output', predict_df)

        df.to_csv("results/" + name[i] + "_results.csv", index=False)


##########################################################################################################
def get_outputs(name, data):
    loaded_model = pickle.load(
        open("models/" + args.model_name + ".sav", 'rb'))

    predictions = loaded_model.predict(data)
    df = pd.DataFrame(data=data)
    df = df.drop(columns=['parse', 'count', 'split', 'original word'])
    df['output'] = predictions
    predict_df = df['output']
    df.drop(labels=['output'], axis=1, inplace=True)
    df.insert(7, 'output', predict_df)

    df.to_csv("results/" + str(name) + "_results.csv", index=False)
##########################################################################################################


def wandB():
    wandb.init(project="fbook_CWI")
    y_data = total_test

    y_test = y_data['complex_binary'].values
    y_pred = pipeline.predict(y_data)
    y_probas = pipeline.predict_proba(y_data)

    vectorizer = CountVectorizer()
    words_test = vectorizer.fit_transform(y_data['word'])

    y_data['word'] = pd.DataFrame(
        words_test.toarray(), columns=vectorizer.get_feature_names())

    X_test = y_data.drop(columns=['complex_binary', 'parse', 'count', 'split', 'original word',
                                  'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic', 'sentence', 'ID', 'clean sentence', 'start_index', 'end_index',  'pos', 'lemma'])

    train = total_training
    words_train = vectorizer.fit_transform(train['word'])
    train['word'] = pd.DataFrame(
        words_train.toarray(), columns=vectorizer.get_feature_names())
    y_train = train['complex_binary'].values
    X_train = train.drop(columns=['complex_binary', 'parse', 'count', 'split', 'original word',
                                  'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic', 'sentence', 'ID', 'clean sentence', 'start_index', 'end_index', 'pos', 'lemma'])

    # temporary, we need to get feature names from the pipeline transformation of features
    # train = X_train.drop(
    #     columns=['sentence', 'ID', 'clean sentence', 'start_index', 'end_index', 'word', 'pos', 'lemma'])
    feature_names = X_train.values

    labels = ['non_complex', 'complex']

    plot_results(pipeline, X_train, X_test, y_train, y_test, y_pred,
                 y_probas, labels, str(model), feature_names)


##########################################################################################################

def plot_results(model, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, model_name, feature_names):
    wandb.sklearn.plot_classifier(model,
                                  X_train, X_test,
                                  y_train, y_test,
                                  y_pred, y_probas,
                                  labels,
                                  False,
                                  model_name,
                                  feature_names)


##########################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--wandb', '-wb', type=int, default=0)
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument('--test', '-t', type=str, default=None)

    parser.add_argument('--model_name', '-mn', type=str, default=None)
    args = parser.parse_args()

    if (args.wikipedia == 1):
        test_name = "wikipedia_test"
        wikipedia_test_data = pd.read_pickle('features/Wikipedia_Test_allInfo')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_test_data.name = 'Wikipedia'
        wikipedia_training_data.name = 'Wikipedia'
        test_frames = [wikipedia_test_data]

    if (args.wikinews == 1):
        test_name = "wikinews_test"
        wiki_test_data = pd.read_pickle('features/WikiNews_Test_allInfo')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_test_data.name = 'WikiNews'
        wiki_training_data.name = 'WikiNews'
        test_frames = [wiki_test_data]

    if (args.news == 1):
        test_name = 'news_test'
        news_test_data = pd.read_pickle('features/News_Test_allInfo')
        news_training_data = pd.read_pickle('features/News_Test_allInfo')
        news_test_data.name = 'News'
        news_training_data.name = 'News'
        test_frames = [news_test_data]

    elif(args.test):
        test_name = args.test
        testing_data = pd.read_pickle('features/' + args.test + '_allInfo')
        testing_data.name = 'testing'
        test_frames = [testing_data]

    total_test = pd.concat(test_frames)

    total_test.fillna(0.0, inplace=True)

    if (args.test):
        get_outputs(args.test, total_test)
    else:
        apply_algorithm(test_name + "_" + args.model_name, [total_test])

    if args.wandb == 1:
        wandB()

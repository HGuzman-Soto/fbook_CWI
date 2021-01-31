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
    loaded_model = pickle.load(
        open("models/" + args.model_name + ".sav", 'rb'))

    if (args.test):
        predict(args.test, loaded_model, [total_test])
    else:
        if (args.predict == 1):
            predict(test_name + "_" + args.model_name,
                    loaded_model, [total_test])
        if (args.evaluation == 1):
            evaluation(test_name + "_" + args.model_name,
                       loaded_model, [total_test])

##########################################################################################################


"""
Evaluation script gives train, test, and later dev results

"""


def evaluation(name, model, array):
    model_stats = pd.DataFrame(
        columns=['Data', 'Classifier', 'Precision', 'Recall', 'F-Score'])
    i = 0
    for x in array:
        print("results for", name, ":\n")
        data = x
        targets = data['complex_binary'].values
        predictions = model.predict(data)

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


def predict(name, model, array):
    i = 0
    for x in array:
        print("predicting for", name)

        data = x
        predictions = model.predict(data)
        df = pd.DataFrame(data=data)

        """
        Messy code down here
        """
        if (args.test):
            df = df.drop(columns=['parse', 'count', 'split', 'original word'])
        else:
            df = df.drop(columns=['parse', 'count', 'split', 'original word',
                                  'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])

        df['output'] = predictions
        predict_df = df['output']
        df.drop(labels=['output'], axis=1, inplace=True)
        df.insert(7, 'output', predict_df)
        df.to_csv("results/" + name + "_results.csv", index=False)

        print("results outputted in results folder", "\n")


##########################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument('--test', '-t', type=str, default=None)
    parser.add_argument('--predict', '-p', type=int, default=0)
    parser.add_argument('--evaluation', '-e', type=int, default=0)
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
    main()

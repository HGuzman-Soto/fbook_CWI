# adapted from: https://github.com/siangooding/cwi_2018/blob/master/Algorithm%20Application.ipynb
##########################################################################################################

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


##########################################################################################################
# import wandb
# wandb.init(project="visualize-sklearn")
# https://docs.wandb.ai/integrations/scikit
# https://colab.research.google.com/drive/1dxWV5uulLOQvMoBBaJy2dZ3ZONr4Mqlo?usp=sharing#scrollTo=Asg7YeGxmJRO

import wandb
from wandb.keras import WandbCallback

# wandb.login()


##########################################################################################################


"""
Todo

1) Think about where/how to save models (wandb + pickle)
2) Right now parser args not super clean. Basically you've got the option to train on all or whichever you want
But the options for which your evaluating is not super clear. My assumption (since camb shows performance best with all datasets)
is that you'll train on all three. So if you choose test, we'll train on all of them

3) Immma add somewhere where the model outputs are saved along with the labels for each dataset

Arguments
1) You can train on all the datasets with -a 1. Or train a subset with args -tw 1, -ti and -tn 1. You can chain
these arguements

2) Then, you test on -w 1, -n 1 or -i 1. You currently can't test on all of them (but it be very easy to change the code
to do that).


"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--all', '-a', type=int, default=0)
    parser.add_argument('--train_wikipedia', '-tw', type=int, default=0)
    parser.add_argument('--train_wikinews', '-ti', type=int, default=0)
    parser.add_argument('--train_news', '-tn', type=int, default=0)
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument('--test', '-t', type=int, default=0)
    parser.add_argument('--random_forest', '-rf', type=int, default=0)
    parser.add_argument('--ada_boost', '-ab', type=int, default=0)
    parser.add_argument('--combine_models', '-cm', type=int, default=0)

    train_frames = []
    test_frames = []
    train_names = []
    args = parser.parse_args()
    if (args.all == 1):
        train_names = ['wikipedia_train', 'wikinews_train', 'news_train']
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_training_data.name = 'Wikipedia'

        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_training_data.name = 'WikiNews'

        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_training_data = news_training_data.drop_duplicates()
        news_training_data.name = 'News'

        train_frames = [wikipedia_training_data,
                        wiki_training_data, news_training_data]

    if (args.train_wikipedia == 1):
        train_names.append('wikipedia_train')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_training_data.name = 'Wikipedia'
        train_frames.append(wikipedia_training_data)

    if (args.train_wikinews == 1):
        train_names.append('wikinews_train')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_training_data.name = 'WikiNews'
        train_frames.append(wiki_training_data)

    if (args.train_news == 1):
        train_names.append('news_train')
        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_training_data.name = 'News'
        train_frames.append(news_training_data)

    if (args.wikipedia == 1):
        test_name = ["wikipedia_test"]
        wikipedia_test_data = pd.read_pickle('features/Wikipedia_Test_allInfo')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_test_data.name = 'Wikipedia'
        wikipedia_training_data.name = 'Wikipedia'
        test_frames = [wikipedia_test_data]

    if (args.wikinews == 1):
        test_name = ["wikinews_test"]
        wiki_test_data = pd.read_pickle('features/WikiNews_Test_allInfo')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_test_data.name = 'WikiNews'
        wiki_training_data.name = 'WikiNews'
        test_frames = [wiki_test_data]

    if (args.news == 1):
        test_name = ['news_test']
        news_test_data = pd.read_pickle('features/News_Test_allInfo')
        news_training_data = pd.read_pickle('features/News_Test_allInfo')
        news_test_data.name = 'News'
        news_training_data.name = 'News'
        test_frames = [news_test_data]

    elif (args.test == 1):
        test_name = ["test"]
        testing_data = pd.read_pickle('features/testing_data_allInfo')
        testing_data.name = 'testing'
        test_frames = [testing_data]

    total_training = pd.concat(train_frames)
    total_test = pd.concat(test_frames)

    total_training.fillna(0.0, inplace=True)
    total_test.fillna(0.0, inplace=True)

    training_data = total_training
    train_targets = training_data['complex_binary'].values


##########################################################################################################


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

    # def get_feature_names(self):
    #     return X.columns.tolist()

##########################################################################################################


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

    # def get_feature_names(self):
    #     return X.columns.tolist()

##########################################################################################################


first_fixation = Pipeline([
    ('selector', NumberSelector(key='IA_FIRST_FIXATION_DURATION')),
    ('standard', StandardScaler())
])

words = Pipeline([
    ('selector', TextSelector(key='word')),
    ('vect', CountVectorizer())
])

word_length = Pipeline([
    ('selector', NumberSelector(key='length')),
    ('standard', StandardScaler())
])

dep_num = Pipeline([
    ('selector', NumberSelector(key='dep num')),
    ('standard', StandardScaler())
])


tag = Pipeline([
    ('selector', TextSelector(key='pos')),
    ('vect', CountVectorizer())
])

synonyms = Pipeline([
    ('selector', NumberSelector(key='synonyms')),
    ('standard', StandardScaler())
])

hypernyms = Pipeline([
    ('selector', NumberSelector(key='hypernyms')),
    ('standard', StandardScaler())
])

hyponyms = Pipeline([
    ('selector', NumberSelector(key='hyponyms')),
    ('standard', StandardScaler())
])

syllables = Pipeline([
    ('selector', NumberSelector(key='syllables')),
    ('standard', StandardScaler())
])

simple_wiki = Pipeline([
    ('selector', NumberSelector(key='simple_wiki')),
    ('standard', StandardScaler())
])

ogden = Pipeline([
    ('selector', NumberSelector(key='ogden')),
    ('standard', StandardScaler())
])


frequency = Pipeline([
    ('selector', NumberSelector(key='google frequency')),
    ('standard', StandardScaler())
])

subimdb = Pipeline([
    ('selector', NumberSelector(key='sub_imdb')),
    ('standard', StandardScaler())
])

cald = Pipeline([
    ('selector', NumberSelector(key='cald')),
    ('standard', StandardScaler())
])


aoa = Pipeline([
    ('selector', NumberSelector(key='aoa')),
    ('standard', StandardScaler())
])
conc = Pipeline([
    ('selector', NumberSelector(key='cnc')),
    ('standard', StandardScaler())
])
fam = Pipeline([
    ('selector', NumberSelector(key='fam')),
    ('standard', StandardScaler())
])
img = Pipeline([
    ('selector', NumberSelector(key='img')),
    ('standard', StandardScaler())
])


KFCAT = Pipeline([
    ('selector', NumberSelector(key='KFCAT')),
    ('standard', StandardScaler())
])

KFSMP = Pipeline([
    ('selector', NumberSelector(key='KFSMP')),
    ('standard', StandardScaler())
])

KFFRQ = Pipeline([
    ('selector', NumberSelector(key='KFFRQ')),
    ('standard', StandardScaler())
])

NPHN = Pipeline([
    ('selector', NumberSelector(key='NPHN')),
    ('standard', StandardScaler())
])

TLFRQ = Pipeline([
    ('selector', NumberSelector(key='TLFRQ')),
    ('standard', StandardScaler())
])


##########################################################################################################

global feats
feats = FeatureUnion([
    ('words', words),
    ('word_length', word_length),
    ('Tag', tag),
    ('dep_num', dep_num),
    ('hypernyms', hypernyms),
    ('hyponyms', hyponyms),
    ('synonyms', synonyms),
    ('Syllables', syllables),
    ('ogden', ogden),
    ('simple_wiki', simple_wiki),
    ('freq', frequency),
    ('subimdb', subimdb),
    ('cald', cald),
    ('aoa', aoa),
    ('cnc', conc),
    ('fam', fam),
    ('img', img),
    ('KFCAT', KFCAT),
    ('KFSMP', KFSMP),
    ('KFFRQ', KFFRQ),
    ('NPHN', NPHN),
    ('TLFRQ', TLFRQ)
])


##########################################################################################################


feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(training_data)

##########################################################################################################

models = []

if (args.ada_boost == 1 or args.combine_models == 1):

    model = AdaBoostClassifier(n_estimators=5000, random_state=67)
    pipeline = Pipeline([
        ('features', feats),
        ('classifier', model),
    ])
    pipeline.fit(training_data, train_targets)

    models.append(pipeline)

if (args.random_forest == 1 or args.combine_models == 1):

    model = RandomForestClassifier(n_estimators=1000)
    pipeline_rf = Pipeline([
        ('features', feats),
        ('classifier', model)
    ])
    pipeline_rf.fit(training_data, train_targets)

    models.append(pipeline_rf)

if (args.combine_models == 1):

    estimators = [('rf', models[1]), ('ada', models[0])]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(training_data, train_targets)
    model = ensemble
    models.append(model)

pipeline = models[-1]

##########################################################################################################


def apply_algorithm(name, array):
    model_stats = pd.DataFrame(
        columns=['Data', 'Classifier', 'Precision', 'Recall', 'F-Score'])

    i = 0
    for x in array:
        print("results for", name[i], ":\n")
        data = x
        targets = data['complex_binary'].values
        predictions = pipeline.predict(data)

        df = pd.DataFrame(data=data)
        df = df.drop(columns=['parse', 'count', 'split', 'original word',
                              'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])
        df['output'] = predictions
        df.to_csv("results/" + name[i] + "_results.csv", index=False)

        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        F_Score = f1_score(targets, predictions, average='macro')

        model_stats.loc[len(model_stats)] = [i, (str(model))[
            :100], precision, recall, F_Score]

        model_stats.to_csv("results/" + name[i] + "_metrics.csv", index=False)
        print("Accuracy", accuracy)
        print("Precision:", model_stats.Precision)
        print("Recall:", model_stats.Recall)
        print("F-Score:", model_stats['F-Score'], "\n")


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
        df['output'] = predictions
        df.to_csv("results/" + name[i] + "_results.csv", index=False)

##########################################################################################################


def wandB():
    wandb.init(project="fbook_CWI")


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
apply_algorithm(test_name, [total_test])
apply_algorithm(train_names, [total_training])


##########################################################################################################

wandB()


y_data = total_test
y_test = y_data['complex_binary'].values
y_pred = pipeline.predict(y_data)
y_probas = pipeline.predict_proba(y_data)
y_train = y_data.drop(columns=['complex_binary', 'parse', 'count', 'split', 'original word',
                               'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])

train = total_training
X_test = train['complex_binary'].values
X_train = train.drop(columns=['complex_binary', 'parse', 'count', 'split', 'original word',
                              'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])


# temporary, we need to get feature names from the pipeline transformation of features
train = X_train.drop(
    columns=['sentence', 'ID', 'clean sentence', 'start_index', 'end_index', 'word', 'pos', 'lemma'])
feature_names = train.values

labels = ['non_complex_word', 'complex_word']

plot_results(pipeline, X_train, X_test, y_train, y_test, y_pred,
             y_probas, labels, str(model), feature_names)

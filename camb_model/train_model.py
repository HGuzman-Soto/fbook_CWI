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
import pickle


"""
Script will train the models only and pickle them

"""

##########################################################################################################


def main():
    feats = feature_extraction()
    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(training_data)
    model = train_model(training_data, feats)
    pickle_model(model)


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


##########################################################################################################


def feature_extraction():
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
        # ('aoa', aoa),
        # ('cnc', conc),
        # ('fam', fam),
        # ('img', img),
        # ('KFCAT', KFCAT),
        # ('KFSMP', KFSMP),
        # ('KFFRQ', KFFRQ),
        # ('NPHN', NPHN),
        # ('TLFRQ', TLFRQ)
    ])
    return feats


def train_model(training_data, feats):

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
    return pipeline


def pickle_model(model):
    pickle.dump(model, open("models/" + args.model_name + ".sav", 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--wandb', '-wb', type=int, default=0)
    parser.add_argument('--all', '-a', type=int, default=0)
    parser.add_argument('--train_wikipedia', '-tw', type=int, default=0)
    parser.add_argument('--train_wikinews', '-ti', type=int, default=0)
    parser.add_argument('--train_news', '-tn', type=int, default=0)
    parser.add_argument('--random_forest', '-rf', type=int, default=0)
    parser.add_argument('--ada_boost', '-ab', type=int, default=0)
    parser.add_argument('--combine_models', '-cm', type=int, default=0)
    parser.add_argument('--model_name', '-mn', type=str, default=None)

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

    if (args.train_news == 1):
        train_names.append('news_train')
        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_training_data.name = 'News'
        train_frames.append(news_training_data)

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

    total_training = pd.concat(train_frames)

    total_training.fillna(0.0, inplace=True)

    training_data = total_training
    train_targets = training_data['complex_binary'].values

    main()

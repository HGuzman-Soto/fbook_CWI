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
from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV
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
import wandb
import pickle


"""
Script will train the models only and pickle them

"""


##########################################################################################################

"""
Either run feature importance or train a model. Since doing both will take twice as long, its best to 
just have a good model already that you want to run feature importance. This means keep track of the name of your model
and what features you used to train it. 

"""


def main():

    if args.feature_importance == 1:
        feats_for_graph = feature_extraction(indice=1)
        model_graph = train_model(training_data, feats_for_graph)
        feature_list = ['pos', 'length', 'vowels', 'syllables', 'dep num', 'synonyms', 'hypernyms',
                        'hyponyms', 'ogden', 'simple_wiki', 'cald', 'cnc', 'img', 'aoa', 'fam',
                        'sub_imdb', 'google frequency', 'KFCAT', 'KFSMP', 'KFFRQ', 'NPHN',
                        'TLFRQ', 'complex_lexicon', 'subtitles_freq', 'wikipedia_freq',
                        'learner_corpus_freq', 'bnc_freq']
        feature_importance(model_graph, feature_list=feature_list)
    else:
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


def feature_extraction(indice=0):
    words = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer())
    ])

    ngram = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer(analyzer='char', ngram_range=(2, 2)))
    ])

    word_length = Pipeline([
        ('selector', NumberSelector(key='length')),
        ('standard', StandardScaler())
    ])

    vowels = Pipeline([
        ('selector', NumberSelector(key='vowels')),
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

    Wikipedia = Pipeline([
        ('selector', NumberSelector(key='wikipedia_freq')),
        ('standard', StandardScaler())
    ])

    BNC = Pipeline([
        ('selector', NumberSelector(key='bnc_freq')),
        ('standard', StandardScaler())
    ])

    lexicon = Pipeline([
        ('selector', NumberSelector(key='complex_lexicon')),
        ('standard', StandardScaler())
    ])

    learners = Pipeline([
        ('selector', NumberSelector(key='learner_corpus_freq')),
        ('standard', StandardScaler())
    ])

    subtitles_corpus = Pipeline([
        ('selector', NumberSelector(key='subtitles_freq')),
        ('standard', StandardScaler())
    ])

    #('ngram', ngram) is omitted
    feature_list = [
    ('words', words),
    ('ngram', ngram),
    ('Tag', tag),
    ('word_length', word_length),
    # ('vowels', vowels),
    ('Syllables', syllables),
    ('dep_num', dep_num),
    ('synonyms', synonyms),
    ('hypernyms', hypernyms),
    ('hyponyms', hyponyms),
    ('ogden', ogden),
    ('simple_wiki', simple_wiki),
    ('cald', cald),
    ('cnc', conc),
    ('img', img),
    ('aoa', aoa),
    ('fam', fam),
    ('subimdb', subimdb),
    ('freq', frequency),
    ('KFCAT', KFCAT),
    ('KFSMP', KFSMP),
    ('KFFRQ', KFFRQ),
    ('NPHN', NPHN),
    ('TLFRQ', TLFRQ),
    # ('complex_lexicon', lexicon),
    # ('subtitles_freq', subtitles_corpus),
    # ('wikipedia_freq', Wikipedia),
    # ('learner_corpus_freq', learners),
    # ('bnc_freq', BNC)
    ]

    feats = FeatureUnion(feature_list[indice:])
    return feats
##########################################################################################################


def grid_search(training_data, feats, model_type):
    if(model_type == "rf"):
        grid = {'classifier__n_estimators': [int(x) for x in np.linspace(start=2000, stop=10000, num=5)],
                'classifier__max_features': ['auto', 'sqrt'],
                'classifier__max_depth': ([int(x) for x in np.linspace(10, 110, num=5)]),
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__bootstrap': [True, False]
                }

        model = RandomForestClassifier()

    if model_type == "ab":
        grid = {'classifier__n_estimators': [int(x) for x in np.linspace(start=2000, stop=20000, num=11)],
                'classifier__learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.1, num=10)]}

        model = AdaBoostClassifier()

    pipeline = Pipeline([
        ('features', feats),
        ('classifier', model)])

    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=grid, cv=3, verbose=2)

    grid_search.fit(training_data, train_targets)

    # format param keys for non-search use
    old = grid_search.best_params_
    params = dict(zip([k[12:] for k in old.keys()], list(old.values())))
    print("OPTIMAL HYPERPARAMETERS:")
    [print(p) for p in params.items()]
    return params

##########################################################################################################


def train_model(training_data, feats):

    models = []

    if (args.ada_boost == 1 or args.combine_models == 1):

        if(args.grid_search == 1):
            model = AdaBoostClassifier(
                **grid_search(training_data, feats, "ab"))
        else:
            model = AdaBoostClassifier(n_estimators=5000, random_state=67)

        pipeline = Pipeline([
            ('features', feats),
            ('classifier', model),
        ])

        if args.recursive_feature == 1:
            rfe = RFE(estimator=model, n_features_to_select=25)
            pipeline = Pipeline([
                ('features', feats),
                ('rfe', rfe),

                ('classifier', model),
            ])
        pipeline.fit(training_data, train_targets)

        models.append(pipeline)

    if (args.random_forest == 1 or args.combine_models == 1):

        if(args.grid_search == 1):
            model = RandomForestClassifier(
                **grid_search(training_data, feats, "rf"))
        else:
            model = RandomForestClassifier(n_estimators=10)

        pipeline_rf = Pipeline([
            ('features', feats),
            ('classifier', model)
        ])

        pipeline_rf.fit(training_data, train_targets)
        feature_importance = pipeline_rf.named_steps['classifier'].feature_importances_

        models.append(pipeline_rf)

    if (args.combine_models == 1):

        estimators = [('rf', models[1]), ('ada', models[0])]
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(training_data, train_targets)
        model = ensemble
        models.append(model)

    pipeline = models[-1]
    return pipeline

##########################################################################################################


"""
Given a pipline with features and a classifier + the names of features
Function plots feature importance 

The words and n-gram features are always ommitted, and the pos-tag will have 30 features. A work 
around was I summed up these first 30 'features' and created a category for pos tags. Then I append
it to the rest of the 24 features.


One area of concern is defining feature list and how to define a susbset of features to train. Defining the feature
list you could just do manually to be honest as you run the code (there's only a few times we will run it). But
how to subset a features when training the model is an ongoing issue.

"""


def feature_importance(pipeline, feature_list):

    pipeline.fit(training_data, train_targets)
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    # also need to denote that word is a feature, same with pos tag (22)
    if (args.all):
        tag_indice = 30
    elif (args.train_news):
        tag_indice = 22

    print(len(feature_importance))
    tag_importance = sum(
        feature_importance[0: tag_indice])

    tag_list = list(tag_importance.flatten())
    final_importance = tag_list + \
        list(feature_importance[tag_indice:].flatten())

    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(
        final_importance, index=feature_list)
    feat_importances.plot(kind='barh')
    plt.show()


##########################################################################################################


def pickle_model(model):
    pickle.dump(model, open("models/" + args.model_name + ".sav", 'wb'))

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
    parser.add_argument('--test', '-t', help="test data for Wandb",
                        type=str, default=None)
    parser.add_argument('--all', '-a', type=int, default=0)
    parser.add_argument('--train_wikipedia', '-tw', type=int, default=0)
    parser.add_argument('--train_wikinews', '-ti', type=int, default=0)
    parser.add_argument('--train_news', '-tn', type=int, default=0)
    parser.add_argument('--train_old', '-to', type=int, default=0)
    parser.add_argument('--random_forest', '-rf', type=int, default=0)
    parser.add_argument('--ada_boost', '-ab', type=int, default=0)
    parser.add_argument('--combine_models', '-cm', type=int, default=0)
    parser.add_argument('--grid_search', '-gs', type=int, default=0)
    parser.add_argument('--recursive_feature', '-rfe', type=int, default=0)
    parser.add_argument('--feature_importance',
                        '-feature', type=int, default=0)
    parser.add_argument(
        '--model_name', '-mn', type=str, default=None)

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
    
    if (args.train_old == 1):
        train_names.append('2016_train')
        old_train_dataset = pd.read_pickle('features/2016_Train_allInfo')
        old_train_dataset.name = '2016_train'
        train_frames.append(old_train_dataset)
    

    total_training = pd.concat(train_frames)

    total_training.fillna(0.0, inplace=True)

    training_data = total_training
    train_targets = training_data['complex_binary'].values

    main()

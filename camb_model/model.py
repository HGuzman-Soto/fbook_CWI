# adapted from: https://github.com/siangooding/cwi_2018/blob/master/Algorithm%20Application.ipynb
##########################################################################################################

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
import string
import numpy as np
import argparse
import pandas as pd
import sys


##########################################################################################################
# import wandb
# wandb.init(project="visualize-sklearn")
# https://docs.wandb.ai/integrations/scikit


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

    train_frames = []
    test_frames = []
    args = parser.parse_args()
    if (args.all == 1):
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
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_training_data.name = 'Wikipedia'
        train_frames.append(wikipedia_training_data)

    if (args.train_wikinews == 1):
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_training_data.name = 'WikiNews'
        train_frames.append(wiki_training_data)

    if (args.train_news == 1):
        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_training_data.name = 'News'
        train_frames.append(news_training_data)

    if (args.wikipedia == 1):
        wikipedia_test_data = pd.read_pickle('features/Wikipedia_Test_allInfo')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_test_data.name = 'Wikipedia'
        wikipedia_training_data.name = 'Wikipedia'
        test_frames = [wikipedia_test_data]

    if (args.wikinews == 1):
        wiki_test_data = pd.read_pickle('features/WikiNews_Test_allInfo')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_test_data.name = 'WikiNews'
        wiki_training_data.name = 'WikiNews'
        test_frames = [wiki_test_data]

    if (args.news == 1):
        news_test_data = pd.read_pickle('features/News_Test_allInfo')
        news_test_data.name = 'News'
        news_training_data.name = 'News'
        test_frames = [news_test_data]

    elif (args.test == 1):
        testing_data = pd.read_pickle('features/testing_data_allInfo')
        testing_data.name = 'testing'
        test_frames = [testing_data]

    # I think this lexicon is in reference to the 2017 wu paper?
    # Or their may be a part that has to do with phrases here
    # lexicon = pd.read_table('lexicon', delim_whitespace=True,
    #                         names=('phrase', 'score'))
    # lexicon['phrase'] = lexicon['phrase'].apply(lambda x: str(x).lower())

    total_training = pd.concat(train_frames)

    total_test = pd.concat(test_frames)

    # total_training = pd.merge(total_training, lexicon, on='phrase', how='left')
    total_training.fillna(0.0, inplace=True)

    # total_test = pd.merge(total_test, lexicon, on='phrase', how='left')
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


first_fixation = Pipeline([
    ('selector', NumberSelector(key='IA_FIRST_FIXATION_DURATION')),
    ('standard', StandardScaler())
])

words = Pipeline([
    ('selector', TextSelector(key='phrase')),
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

# n_gram_freq = Pipeline([
#     ('selector', NumberSelector(key='ngram freq')),
#     ('standard', StandardScaler())
# ])

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

# score = Pipeline([
#     ('selector', NumberSelector(key='score')),
#     ('standard', StandardScaler())
# ])

##########################################################################################################

global feats
feats = FeatureUnion([  # ('ff',first_fixation),
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
    # ('origin', origin),
    ('freq', frequency),
    ('subimdb', subimdb),
    # ('n_gram_freq', n_gram_freq),
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
    # ('score', score)
])


##########################################################################################################


feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(training_data)

##########################################################################################################


model = AdaBoostClassifier(n_estimators=5000, random_state=67)
pipeline = Pipeline([
    ('features', feats),
    ('classifier', model),
])

pipeline.fit(training_data, train_targets)


"""

Ensemble method - 
"""
# rf = RandomForestClassifier(n_estimators=1000)

# pipeline_rf = Pipeline([
#     ('features', feats),
#     ('classifier', rf)
# ])
# pipeline_rf.fit(training_data, train_targets)


# estimators = [('rf', pipeline_rf), ('ada', pipeline)]
# ensemble = VotingClassifier(estimators, voting='hard')
# ensemble.fit(training_data, train_targets)

##########################################################################################################

global model_stats
model_stats = pd.DataFrame(
    columns=['Data', 'Classifier', 'Precision', 'Recall', 'F-Score'])


def apply_algorithm(array):

    i = 0
    for x in array:

        test_data = x
        test_targets = test_data['complex_binary'].values
        # test_predictions = ensemble.predict(test_data)
        test_predictions = pipeline.predict(test_data)

        accuracy = accuracy_score(test_targets, test_predictions)
        precision = precision_score(test_targets, test_predictions)
        recall = recall_score(test_targets, test_predictions)
        F_Score = f1_score(test_targets, test_predictions, average='macro')

        model_stats.loc[len(model_stats)] = [i, (str(model))[
            :100], precision, recall, F_Score]
        print("Accuracy", accuracy)
        print("Precision:", model_stats.Precision)
        print("Recall:", model_stats.Recall)
        print("F-Score:", model_stats['F-Score'])

        # baseline_accuracies(test_targets)

##########################################################################################################


def fbook(fbook_data, features_df):
    test_predictions = pipeline.predict(fbook_data)
    outputs_df = pd.DataFrame(data=test_predictions)
    final_df = features_df.concat(outputs_df)
    final_df.to_csv('testing_results.csv', index=False)

##########################################################################################################


apply_algorithm([total_test])  # with lexicon
print(model_stats)

if (args.test == 1):
    fbook(total_training, test_df)


##########################################################################################################

"""
Below is what is needed to run the model on our generated fbook data
Ignore for now as we're not working with that at this moment right now

"""

# fbook_data = pd.read_pickle('features/test_data')
# fbook_data.to_csv('testing_features.csv', index=False)
# print(fbook_data.head())
# fbook(fbook_data)


# apply_algorithm([total_test])  # without lexicon
# model_stats

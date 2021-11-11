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
from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from boruta import BorutaPy
import itertools
import csv


import string
import numpy as np
import argparse
import pandas as pd
import sys
import gensim
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
        feats_for_graph = feature_extraction(indice=0)
        model_graph = train_model(training_data, feats_for_graph)

        feature_list = []

        if(args.spanish):
            """feature_list += ['word','length','syllables','synonyms','vowels','wikipedia_freq','learner_corpus_freq','subtitles_freq','news_freq','google_freq','simple_wiki_fourgram','simple_wiki_bigrams']
            """
            print("adding spanish features")
            feature_list += ['word','wikipedia_freq','learners_freq','subtitles_freq','news_freq','google_freq','simple_wiki_bigrams','wimple_wiki_fourgram','length','syllables','vowels','synonyms']
        else:
            feature_list = ['pos', 'simple_wiki_bigrams', 'learners_bigrams', 'google_char_bigram', 'google_char_trigram', 'simple_wiki_fourgram', 'learner_fourgram',
                            'length', 'vowels', 'syllables', 'consonants', 'dep num', 'synonyms', 'hypernyms',
                            'hyponyms', 'holonyms', 'meronyms', 'ogden', 'ner', 'simple_wiki', 'cald', 'cnc', 'img', 'aoa', 'fam',
                            'sub_imdb', 'google frequency', 'KFCAT', 'KFSMP', 'KFFRQ', 'NPHN',
                            'TLFRQ', 'complex_lexicon', 'subtitles_freq', 'wikipedia_freq',
                            'learner_corpus_freq', 'bnc_freq']
        
        

        feature_importance(model_graph, feature_list=feature_list)

    elif args.recursive_feature:
        recursive_feat(args.recursive_feature)

    elif args.ovl:
        OVL()

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


# ##########################################################################################################
# # model = gensim.models.KeyedVectors.load_word2vec_format("dewiki_20180420_100d.txt")
# model = gensim.models.KeyedVectors.load_word2vec_format("german.model", binary=True)

# class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, key):
#         self.key = key

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X_ = X.copy()
#         vecs = []

#         for i in range(0, len(X_)):
#             try:
#                 vecs.append(model[X_[i]])
#             except:
#                 vecs.append(np.array(np.zeros(100)))

#         vecs = np.array(vecs)
#         return vecs




# ##########################################################################################################



def feature_extraction(indice=0):

    feature_list = []
    pipe_feats = []

    words = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer())
    ])

    bi_gram_char = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)))
    ])

    four_gram_char = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(4, 4)))
    ])

    simple_wiki_bigrams = Pipeline([
        ('selector', NumberSelector(key='simple_wiki_bigrams')),
        ('standard', StandardScaler())
    ])

    learners_bigrams = Pipeline([
        ('selector', NumberSelector(key='learners_bigrams')),
        ('standard', StandardScaler())
    ])

    google_char_bigram = Pipeline([
        ('selector', NumberSelector(key='google_char_bigram')),
        ('standard', StandardScaler())
    ])

    google_char_trigram = Pipeline([
        ('selector', NumberSelector(key='google_char_trigram')),
        ('standard', StandardScaler())
    ])

    simple_wiki_fourgram = Pipeline([
        ('selector', NumberSelector(key='simple_wiki_fourgram')),
        ('standard', StandardScaler())
    ])

    learner_fourgram = Pipeline([
        ('selector', NumberSelector(key='learner_fourgram')),
        ('standard', StandardScaler())
    ])

    word_length = Pipeline([
        ('selector', NumberSelector(key='length')),
        ('standard', StandardScaler())
    ])

    vowels = Pipeline([
        ('selector', NumberSelector(key='vowels')),
        ('standard', StandardScaler())
    ])

    consonants = Pipeline([
        ('selector', NumberSelector(key='consonants')),
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

    holonyms = Pipeline([
        ('selector', NumberSelector(key='holonyms')),
        ('standard', StandardScaler())
    ])

    meronyms = Pipeline([
        ('selector', NumberSelector(key='meronyms')),
        ('standard', StandardScaler())
    ])

    syllables = Pipeline([
        ('selector', NumberSelector(key='syllables')),
        ('standard', StandardScaler())
    ])

    is_entity = Pipeline([
        ('selector', NumberSelector(key='ner')),
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
        ('selector', NumberSelector(key='learners_freq')),
        ('standard', StandardScaler())
    ])

    subtitles_corpus = Pipeline([
        ('selector', NumberSelector(key='subtitles_freq')),
        ('standard', StandardScaler())
    ])

    #embeddings

    # for x in range(1,301):
    #     key = 'embed_' + str(x)

    #     embed = Pipeline([
    #     ('selector', NumberSelector(key=key)),
    #     ('standard', StandardScaler())
    #     ])

    #     feature_list.append((key, embed))
    #     pipe_feats.append((key, embed))

    ###################################################################################
    # adding in German features
    # define some features again for correct feat names

    pos = Pipeline([
        ('selector', TextSelector(key='pos')),
        ('vect', CountVectorizer())
    ])

    ner = Pipeline([
        ('selector', NumberSelector(key='ner')),
        ('standard', StandardScaler())
    ])

    learners = Pipeline([
        ('selector', NumberSelector(key='learners_freq')),
        ('standard', StandardScaler())
    ])

    news = Pipeline([
        ('selector', NumberSelector(key='news_freq')),
        ('standard', StandardScaler())
    ])

    frequency = Pipeline([
        ('selector', NumberSelector(key='google_freq')),
        ('standard', StandardScaler())
    ])

    wiki_char_bigram = Pipeline([
        ('selector', NumberSelector(key='wiki_char_bigram')),
        ('standard', StandardScaler())
    ])

    wiki_char_fourgram = Pipeline([
        ('selector', NumberSelector(key='wiki_char_fourgram')),
        ('standard', StandardScaler())
    ])

    bi_gram_char = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)))
    ])

    four_gram_char = Pipeline([
        ('selector', TextSelector(key='word')),
        ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(4, 4)))
    ])

# feature_list += ['word','length','syllables','synonyms','vowels','pos','wikipedia_freq','learner_corpus_freq','subtitles_freq','news_freq','google_freq','simple_wiki_fourgram','simple_wiki_bigrams']
    # German feature list
    feature_list += [
        ('word', words),
        ('wikipedia_freq', Wikipedia),
        ('learners_freq', learners),
        ('subtitles_freq', subtitles_corpus),
        ('news_freq', news),
        ('pos', pos),
        ('google_freq', frequency),
        ('simple_wiki_bigrams', bi_gram_char),
        ('simple_wiki_fourgram', four_gram_char),
        ('length', word_length),
        ('syllables', syllables),
        ('vowels', vowels),
        ('synonyms', synonyms)
    ]


    # #('ngram', ngram) is omitted
    # feature_list += [
    #     # ('words', words),
    #     # ('bigram_char', bi_gram_char),
    #     # ('four_gram_char', four_gram_char),
    #     ('Tag', tag),
    #     ('simple_wiki_bigrams', simple_wiki_bigrams),
    #     ('learners_bigrams', learners_bigrams),
    #     ('google_char_bigram', google_char_bigram),
    #     ('google_char_trigram', google_char_trigram),
    #     ('simple_wiki_fourgram', simple_wiki_fourgram),
    #     ('learner_fourgram', learner_fourgram),
    #     ('word_length', word_length),
    #     ('vowels', vowels),
    #     ('consonants', consonants),
    #     ('Syllables', syllables),
    #     ('dep_num', dep_num),
    #     ('synonyms', synonyms),
    #     ('hypernyms', hypernyms),
    #     ('hyponyms', hyponyms),
    #     ('holonyms', holonyms),
    #     ('meronyms', meronyms),
    #     ('ogden', ogden),
    #     ('ner', is_entity),
    #     ('simple_wiki', simple_wiki),
    #     ('cald', cald),
    #     ('cnc', conc),
    #     ('img', img),
    #     ('aoa', aoa),
    #     ('fam', fam),
    #     ('subimdb', subimdb),
    #     ('freq', frequency),
    #     ('KFCAT', KFCAT),
    #     ('KFSMP', KFSMP),
    #     ('KFFRQ', KFFRQ),
    #     ('NPHN', NPHN),
    #     ('TLFRQ', TLFRQ),
    #     ('complex_lexicon', lexicon),
    #     ('subtitles_freq', subtitles_corpus),
    #     ('wikipedia_freq', Wikipedia),
    #     ('learner_corpus_freq', learners),
    #     ('bnc_freq', BNC)
    # ]

    if (args.feature_importance == 1):
        feats = FeatureUnion(feature_list[indice:])

    pipe_feats += [
        # ('words', words),
        # ('bigram_char', bi_gram_char),
        # ('four_gram_char', four_gram_char),
        #('Tag', tag),
        ('simple_wiki_bigrams', bi_gram_char),
        ('learners_bigrams', learners_bigrams),
        ('google_char_bigram', google_char_bigram),
        ('google_char_trigram', google_char_trigram),
        ('simple_wiki_fourgram', four_gram_char),
        ('learner_fourgram', learner_fourgram),
        ('length', word_length),
        ('vowels', vowels),
        ('syllables', syllables),
        ('dep_num', dep_num),
        ('synonyms', synonyms),
        ('hypernyms', hypernyms),
        ('hyponyms', hyponyms),
        ('holonyms', holonyms),
        ('meronyms', meronyms),
        ('ogden', ogden),
        ('pos', pos),
        ('ner', is_entity),
        ('simple_wiki', simple_wiki),
        ('cald', cald),
        ('cnc', conc),
        ('img', img),
        ('aoa', aoa),
        ('fam', fam),
        ('subimdb', subimdb),
        ('google_freq', frequency),
        ('KFCAT', KFCAT),
        ('KFSMP', KFSMP),
        ('KFFRQ', KFFRQ),
        ('NPHN', NPHN),
        ('TLFRQ', TLFRQ),
        ('complex_lexicon', lexicon),
        ('news_freq', news),
        ('subtitles_freq', subtitles_corpus),
        ('wikipedia_freq', Wikipedia),
        ('learners_freq', learners),
        ('bnc_freq', BNC)

    ]

    if args.features:
        feats_in = args.features.strip("[]").split(",")
        pipe_feats = [x for x in pipe_feats if x[0] in feats_in]
        print("Selected # of features: " + str(len(pipe_feats)))
        print("Features: ")
        print(pipe_feats)
        feats = FeatureUnion(pipe_feats)
    else:
        feats = FeatureUnion(feature_list)
    print(feats)
    return feats
##########################################################################################################


def grid_search(training_data, feats, model_type):
    if(model_type == "rf"):
        grid = {'classifier__n_estimators': [2000, 5000, 10000],
                'classifier__max_features': ['auto', 'sqrt'],
                'classifier__max_depth': (10, 25, 50, 75, 100),
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__bootstrap': [True, False]
                }

        model = RandomForestClassifier()

    if model_type == "ab":
        grid = {'classifier__n_estimators': [int(x) for x in np.linspace(start=4000, stop=6000, num=3)],
                'classifier__learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.1, num=5)]}

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
        pipeline.fit(training_data, train_targets)
        models.append(pipeline)

    if (args.random_forest == 1 or args.combine_models == 1):

        if(args.grid_search == 1):
            model = RandomForestClassifier(
                **grid_search(training_data, feats, "rf"))
        else:
            model = RandomForestClassifier(n_estimators=5000)

        pipeline_rf = Pipeline([
            ('features', feats),
            ('classifier', model)
        ])

        pipeline_rf.fit(training_data, train_targets)

        models.append(pipeline_rf)

    if (args.combine_models == 1):

        estimators = [('rf', models[1]), ('ada', models[0])]
        ensemble = VotingClassifier(estimators, voting='soft')
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

    #testing
    tag_indice = 5
    print(feature_importance)

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

"""
Implementation of Recursive Feature Elimination
"""


def recursive_feat(argVal):

    if args.ada_boost == 1:
        model = AdaBoostClassifier(n_estimators=5000)
    else:
        model = RandomForestClassifier(n_estimators=5000)

    useful_data = training_data.drop(['sentence', 'ID', 'start_index', 'end_index',
                                      'word', 'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic',
                                      'split', 'count', 'word', 'pos'], axis=1)

    if(argVal == 1):
        rfecv = RFECV(model, n_jobs=6, verbose=1)
        pipeline = Pipeline([
            ('rfecv', rfecv),
            ('classifier', model),
        ])

        pipeline.fit(useful_data, train_targets)

        print("optimal # of features: %d" % rfecv.n_features_)
        print("used features:")
        best_feats = [useful_data.columns[x]
                      for x in range(len(useful_data.columns)) if rfecv.support_[x]]
        for f in best_feats:
            print(f)

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (fraction of correct classifications)")
        plt.plot(rfecv.grid_scores_)
        plt.show()

        print("\n", rfecv.grid_scores_)

    # else, find best n-features combination
    else:
        # this is where the magic happens
        if(argVal > 1):
            print("Looking for best combination with " +
                  str(argVal) + " features")

            rfe = RFE(model, n_features_to_select=argVal, verbose=1)
            pipeline = Pipeline([
                ('rfe', rfe),
                ('classifier', model),
            ])

            pipeline.fit(useful_data, train_targets)

            print("ranking")
            print(rfe.ranking_)
            print("support: ")
            print(rfe.support_)

            print("Best Combination of " + str(argVal) + " features: ")
            best_feats = [useful_data.columns[x]
                          for x in range(len(useful_data.columns)) if rfe.support_[x]]
            print(best_feats)


##########################################################################################################


"""
Implementation of OVL Feature Selection
"""


def OVL():

    # new random state
    rng = np.random.RandomState(42)

    train_cleaned = training_data.drop(['sentence', 'ID', 'start_index', 'end_index',
                                        'word', 'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic',
                                        'split', 'count', 'word', 'pos'], axis=1)

    # for each feature
    feature_set = []
    for f in train_cleaned.columns:

        # get numpy arrays for feat rows -> true, feat rows -> false
        x0 = np.array([y for x, y in enumerate(
            train_cleaned[f]) if train_targets[x] == 0])
        x1 = np.array([y for x, y in enumerate(
            train_cleaned[f]) if train_targets[x] == 1])

        # make kde plots
        kde0 = stats.gaussian_kde(x0, bw_method=0.3)
        kde1 = stats.gaussian_kde(x1, bw_method=0.3)

        # make x boundaries
        xmin = min(x0.min(), x1.min())
        xmax = min(x0.max(), x1.max())
        # add a 20% margin, as the kde is wider than the data
        dx = 0.2 * (xmax - xmin)
        xmin -= dx
        xmax += dx

        # find minimum of kde intersect for 500 x-values
        x = np.linspace(xmin, xmax, 500)
        kde0_x = kde0(x)
        kde1_x = kde1(x)
        inters_x = np.minimum(kde0_x, kde1_x)

        # plot kde's and intersect
        plt.plot(x, kde0_x, color='b', label='non-complex')
        plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
        plt.plot(x, kde1_x, color='orange', label='complex')
        plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
        plt.plot(x, inters_x, color='r')
        plt.fill_between(x, inters_x, 0, facecolor='none',
                         edgecolor='r', hatch='xx', label='intersection')

        # get area of intersect
        area_inters_x = np.trapz(inters_x, x)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels[2] += f': {area_inters_x * 100:.1f} %'

        # show plot
        # plt.legend(handles, labels, title='Complex?')
        # plt.xlabel(f)
        # plt.ylabel("Probability Density")
        # plt.show()
        # save_path = 'OVLResults/all/all_' + f + '.png'
        # plt.savefig(save_path)
        # plt.cla()
        tup = (f, area_inters_x)
        feature_set.append(tup)

    feature_set = sorted(feature_set, key=lambda x: x[1])

    ##
    # show bar plot of feature names and performance
    ##

    names = list(zip(*feature_set))[0]
    performance = list(zip(*feature_set))[1]
    x_pos = np.arange(len(names))

    plt.cla()
    plt.bar(x_pos, performance, align='center')
    plt.title('Complex/Non-complex Feature Value KDE Plot Overlap')
    plt.xlabel('Feature name')
    plt.ylabel('Percentage of Overlap')
    plt.xticks(x_pos, names, rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.show()


##########################################################################################################

def clean_data():
    training_data = training_data.drop(['sentence', 'ID', 'clean sentence', 'parse', 'start_index', 'end_index',
                                        'word', 'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic',
                                        'split', 'count', 'word', 'original word', 'lemma', 'pos'], axis=1)


##########################################################################################################


def pickle_model(model):
    print("pickling model")
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
    parser.add_argument('--dev_wikipedia', '-dw', type=int, default=0)
    parser.add_argument('--dev_wikinews', '-di', type=int, default=0)
    parser.add_argument('--news', '-dn', type=int, default=0)
    parser.add_argument('--random_forest', '-rf', type=int, default=0)
    parser.add_argument('--ada_boost', '-ab', type=int, default=0)
    parser.add_argument('--combine_models', '-cm', type=int, default=0)
    parser.add_argument('--grid_search', '-gs', type=int, default=0)
    parser.add_argument('--recursive_feature', '-rfe', type=int, default=0)
    parser.add_argument('--boruta', '-bor', type=int, default=0)
    parser.add_argument('--ovl', '-ovl', type=int, default=0)
    parser.add_argument('--features', '-fp', type=str, default="")
    parser.add_argument('--feature_importance',
                        '-feature', type=int, default=0)
    parser.add_argument(
        '--model_name', '-mn', type=str, default=None)

    ## language args
    parser.add_argument('--spanish', '-sp', type=int, default=0)


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
        news_training_data = news_training_data[:50]
        news_training_data.to_csv('test.csv')
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

    elif (args.dev_wikipedia == 1):
        train_names.append('wikipedia_dev')
        wikipedia_dev_data = pd.read_pickle(
            'features/Wikipedia_Dev_allInfo')
        wikipedia_dev_data.name = 'Wikipedia'
        train_frames.append(wikipedia_dev_data)

    elif (args.dev_wikinews == 1):
        train_names.append('WikiNews_dev')
        wikinews_dev_data = pd.read_pickle(
            'features/WikiNews_Dev_allInfo')
        wikinews_dev_data.name = 'WikiNews'
        train_frames.append(wikinews_dev_data)

    elif (args.news == 1):
        train_names.append('news_dev')
        news_dev_data = pd.read_pickle(
            'features/News_Dev_allInfo')
        news_dev_data.name = 'News'
        train_frames.append(news_dev_data)

    ##work on pickling the data
    elif (args.spanish == 1):
        train_names.append('Spanish_train')
        spanish_train_data = pd.read_pickle('features/Spanish_Train_allInfo')
        spanish_train_data.name = 'Spanish'
        train_frames.append(spanish_train_data)


    total_training = pd.concat(train_frames)

    total_training.fillna(0.0, inplace=True)

    training_data = total_training
    train_targets = training_data['complex_binary'].values

    main()
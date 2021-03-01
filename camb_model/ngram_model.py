# code courtesy of https://nlpforhackers.io/language-models/
from pandas import pd
import pickle
from nltk import bigrams, trigrams
from nltk.tokenize import sent_tokenize

from collections import Counter, defaultdict



def create_model(corpus_df, corpus_name):
    # Create a placeholder for model
    model = defaultdict(lambda: defaultdict(lambda: 0))

    corpus_sentences = sent_tokenize(corpus_df)
    # Count frequency of co-occurance  
    for sentence in corpus_sentences:
        for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
            model[(w1)][w2] += 1
    
    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    
    pickle.dump(model, open("lm/" + corpus_name + ".sav", 'wb'))

    
    
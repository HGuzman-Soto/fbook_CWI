import pandas as pd
import argparse
import pickle
##########################################################################################################

"""
Given a dataset which contains features, and a name, the function outputs features.csv

"""


def get_features(data, name):
    df = pd.DataFrame(data=data)
    df = df.drop(columns=['parse', 'count', 'split', 'original phrase',
                          'total_native', 'total_non_native', 'native_complex', 'non_native_complex', 'complex_probabilistic'])
    df.to_csv('features/' + name + '_features.csv',
              index=False, encoding='utf-8-sig')
    return df


##########################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument('--test', '-t', type=int, default=0)

    args = parser.parse_args()

    if (args.wikipedia == 1):
        wikipedia_test_data = pd.read_pickle('features/Wikipedia_Test_allInfo')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_test_data.name = 'Wikipedia'
        wikipedia_training_data.name = 'Wikipedia'
        get_features(wikipedia_training_data, "wikipedia_train")
        get_features(wikipedia_test_data, "wikipedia_test")

    if (args.wikinews == 1):
        wiki_test_data = pd.read_pickle('features/WikiNews_Test_allInfo')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_test_data.name = 'WikiNews'
        wiki_training_data.name = 'WikiNews'
        get_features(wiki_training_data, "wikiNews_train")
        get_features(wiki_test_data, "wikiNews_test")

    if (args.news == 1):
        news_test_data = pd.read_pickle('features/News_Test_allInfo')
        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_test_data.name = 'News'
        news_training_data.name = 'News'
        get_features(news_training_data, "news_train")
        get_features(news_test_data, "news_test")

    elif (args.test == 1):
        testing_data = pd.read_pickle('features/testing_data_allInfo')
        testing_data.name = 'testing'
        test_df = get_features(testing_data, "test")

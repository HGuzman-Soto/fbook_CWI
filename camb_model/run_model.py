import scipy.stats as stats
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import pandas as pd
import pickle
import wandb
from wandb.keras import WandbCallback
from train_model import TextSelector, NumberSelector


def main():
    loaded_model = pickle.load(
        open("models/" + args.model_name + ".sav", 'rb'))

    if (args.test):
        predict(args.test, loaded_model, test_frames)
    else:
        if (args.predict == 1):
            predict(test_name + "_" + args.model_name,
                    loaded_model, test_frames)
        if (args.evaluation == 1):
            evaluation(test_name + "_" + args.model_name,
                       loaded_model, test_frames)

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

        model_stats.loc[len(model_stats)] = [
            i, (str(model)), precision, recall, F_Score]
        print("Accuracy", accuracy)
        print("Precision:", model_stats.Precision)
        print("Recall:", model_stats.Recall)
        print("F-Score:", model_stats['F-Score'], "\n")
        model_stats.to_csv('results/metrics/' + name +
                           "_metrics.csv", index=False)

##########################################################################################################


def predict(name, model, array):
    i = 0
    arr = ['test', 'train']
    for x in array:
        print("predicting for", name + "_" + arr[i])

        data = x
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        df = pd.DataFrame(data=data)

        """
        Messy code down here
        """

        df = df.drop(columns=['parse', 'count', 'split', 'original word'])

        df['output'] = predictions
        df['probability'] = probabilities

        predict_df, probab_df = df['output'], df['probability']
        df.drop(labels=['output', 'probability'], axis=1, inplace=True)
        df.insert(7, 'output', predict_df)
        df.insert(8, 'probability', probab_df)

        df.to_csv("results/" + name + "_" +
                  arr[i] + "_results.csv", index=False)

        print("results outputted in results folder", "\n")
        i += 1

##########################################################################################################


def parse_all_args():
    parser = argparse.ArgumentParser(description='Run feature extraction')
    parser.add_argument('--wikipedia', '-w', type=int, default=0)
    parser.add_argument('--wikinews', '-i', type=int, default=0)
    parser.add_argument('--news', '-n', type=int, default=0)
    parser.add_argument(
        '--test', '-t', help="name of test file", type=str, default=None)
    parser.add_argument('--features', '-f', type=str, default="")
    parser.add_argument('--predict', '-p', type=int, default=1)
    parser.add_argument('--evaluation', '-e', type=int, default=1)
    parser.add_argument('--model_name', '-mn', type=str, default=None)

    args = parser.parse_args()

    return args


##########################################################################################################
if __name__ == "__main__":
    args = parse_all_args()

    if (args.wikipedia == 1):
        test_name = "wikipedia"
        wikipedia_test_data = pd.read_pickle('features/Wikipedia_Test_allInfo')
        wikipedia_training_data = pd.read_pickle(
            'features/Wikipedia_Train_allInfo')
        wikipedia_test_data.name = 'Wikipedia'
        wikipedia_training_data.name = 'Wikipedia'
        test_frames = [wikipedia_test_data, wikipedia_training_data]

    if (args.wikinews == 1):
        test_name = "wikinews"
        wiki_test_data = pd.read_pickle('features/WikiNews_Test_allInfo')
        wiki_training_data = pd.read_pickle('features/WikiNews_Train_allInfo')
        wiki_test_data.name = 'WikiNews'
        wiki_training_data.name = 'WikiNews'
        test_frames = [wiki_test_data, wiki_training_data]

    if (args.news == 1):
        test_name = 'news'
        news_test_data = pd.read_pickle('features/News_Test_allInfo')
        news_training_data = pd.read_pickle('features/News_Train_allInfo')
        news_test_data.name = 'News'
        news_training_data.name = 'News'
        test_frames = [news_test_data, news_training_data]

    elif(args.test):
        test_name = args.test
        testing_data = pd.read_pickle('features/' + args.test + '_allInfo')
        testing_data.name = 'testing'
        test_frames = [testing_data]
    
    if (args.features):
        used_feats = args.features.strip("[]").split(",")
        print("it is:")
        print(used_feats)
        for i in range(len(test_frames)):
            print(test_frames[i])
            test_frames[i]= test_frames[i][used_feats]
            print(test_frames[i])

    # total_test = pd.concat(test_frames)
    # total_test.fillna(0.0, inplace=True)
    main()

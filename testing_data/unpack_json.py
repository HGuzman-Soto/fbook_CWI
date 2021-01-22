"""
Script takes json objects from chrome extension and organizes the data into a csv files

"""
from pathlib import Path
from os import path
import re
import shutil
import os.path
import argparse
import csv
import ast
import pandas as pd
import json
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt') run once


"""
To add to existing json file, use the argument --a 1
To retrieve json file, use the argument --j 1

"""


def get_jsonfile():
    for files in os.listdir():
        if files.endswith('.json'):
            return files


def find_jsonfile():
    path_to_download_folder = str(os.path.join(Path.home(), "Downloads"))
    RootDir1 = path_to_download_folder
    TargetFolder = os.getcwd()

    for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):

        max_num = 0
        max_name = ''
        matches = ['.json', 'threads']
        for name in files:
            # base case
            if all(x in name for x in matches):

                # # copies file to target folder
                current_digit = re.findall(r'\d+', name)

                # check that a digit exists - else turn to int
                if not current_digit:
                    continue
                else:
                    current_digit = int(current_digit[0])

                if (current_digit > max_num):
                    max_num = current_digit
                    max_name = name

    if (max_num == 0):
        max_name = 'threads.json'

    SourceFolder = os.path.join(root, max_name)
    shutil.copy2(SourceFolder, TargetFolder)


def main():
    json_file = get_jsonfile()
    if json_file in 'json_files/':
        json_file = 'json_files/' + json_file

    # vectorized data operations

        d = pd.read_json(json_file, orient='DataFrame')
        shutil.move(json_file,
                    "json_files/")
    else:
        d = pd.read_json(json_file, orient='DataFrame')

    df = pd.json_normalize(d['threads'])

    # drop comment id from df
    df = df.drop('commentid', axis=1)

    # remove post column as seperate df
    df_post = df[['post']].copy()
    df = df.drop('post', axis=1)

    # get post index as id_1
    df['id_1'] = list(df.index.values)

    # get number of comments in each line as range(), move them to seperate df and re-add as id_2 (comment indexes)
    #     once comment indexes are exploded.
    df['id_2'] = df['comments'].apply(lambda x: range(1, len(x)+1))
    df_temp = df[['id_2']].copy()
    df = df.explode('comments', ignore_index=True)
    df_temp = df_temp.explode('id_2', ignore_index=True)
    df['id_2'] = df_temp[['id_2']].copy()

    # run sent_tokenize on 'comments' then make indexes for the text as id_3
    df['comments'] = df['comments'].apply(lambda x: sent_tokenize(str(x)))
    df['id_3'] = df['comments'].apply(lambda x: range(1, len(x)+1))
    df_temp = df[['id_3']].copy()
    df = df.explode('comments', ignore_index=True)
    df_temp = df_temp.explode('id_3', ignore_index=True)
    df['id_3'] = df_temp[['id_3']].copy()

    # merge id's
    df['id'] = df.apply(lambda x: (str(x['id_1']) + "_" +
                                   str(x['id_2']) + "_" + str(x['id_3'])), axis=1)

    # drop id cols and rename column
    df = df.drop(['id_1', 'id_2', 'id_3'], axis=1)
    df = df.rename(columns={'comments': 'text'})

    # work on posts, similarly as with comments

    df_post['post'] = df_post['post'].apply(lambda x: sent_tokenize(str(x)))
    df_post['id_1'] = range(0, len(df_post))
    df_post["id_3"] = df_post['post'].apply(lambda x: range(len(x)))
    df_temp = df_post[['id_3']].copy()
    df_temp = df_temp.explode('id_3', ignore_index=True)
    df_post = df_post.explode('post', ignore_index=True)
    df_post['id_3'] = df_temp[['id_3']].copy()
    df_post['id'] = df_post.apply(lambda x: (
        str(x['id_1']) + "_0_" + str(x['id_3'])), axis=1)
    df_post = df_post.drop(['id_1', 'id_3'], axis=1)
    df_post = df_post.rename(columns={'post': 'text'})

    # bring all data together, remove list bracket leftovers, drop duplicates and empty text
    df = df.append(df_post, ignore_index=True)
    df = df.astype(str)
    df = df[df['text'] != ("" or "nan")]
    df['text'] = df['text'].apply(lambda x: re.sub(
        r"\W*\[\"|\W*\[\'|\'\]\W*|\"\]*\W|\]|\[|^\"|\"$", "", x))
    df.drop_duplicates(subset=['text'], keep='last',
                       inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    # df to csv

    if path.exists('temp_data.csv'):
        df.to_csv('temp_data.csv', mode='a', header=False, index=False)
        # ensures that duplicate comments are dropped from csv
        newdf = pd.read_csv('temp_data.csv')
        newdf.drop_duplicates(
            subset=['text'], keep='first', inplace=True, ignore_index=True)
        newdf.to_csv('temp_data.csv', index=False)
    else:
        df.to_csv('temp_data.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize json data')
    parser.add_argument('--json', '--j', type=int, default=0)

    args = parser.parse_args()
    if (args.json == 1):
        find_jsonfile()
    main()

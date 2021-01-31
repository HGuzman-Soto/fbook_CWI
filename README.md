# Facebook Complex Word Identification

Developed as a part of Dr. Yudong Liu research lab (https://cs.wwu.edu/liuy2) </br>
Currently a work in progress

## Installation

1. Run pip3 install -r requirements.txt to get all packages\
2. When running feature extraction, you'll need to have nlp-core running. Here is an installation guide: https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/

## Instructions For testing data

### 1. Run python3 unpack_json.py to unpack json object into proper data format (text, id)

Note: you can either <br> 1. manually place the json file in the directory and run unpack_json.py <br> 2. or run unpack_json.py with -j command to automatically retrieve the last
downloaded json file from the scraper

ex: python3 unpack_json.py -j 1

### 2. Next, run.py

This will run preprocessing on the text and extract content words using POS tagging and NER.
It will also attach the starting and ending indexes of each content word

ex: python3 run.py

### Columns are identified as such:

- sentence - raw social media text. Includes post + comment
- id - postID_commentID_sentenceID
- clean_sentence - preprocessed text
- word - extracted content word
- start_index - beginning index of the content word, from the cleaned text
- end_index - ending index of the content word, from the cleaned text

##### Diagram of Logical Flow

![GitHub Logo](diagrams/pipeline_high.png)

#### Diagram of Pipeline Architecture

![GitHub Logo](diagrams/pipeline_overview.png)

## Instructions for Camb_model

Make sure to unzip the features.zip and results.zip file

### 1. Run feature_extracton.py on the test set with the following argument:

python3 feature_extraction.py -t {name of data file}

ex: python3 feature_extraction.py -t data_1

### 2. To train a model (train_model.py), these are the following instructions:

You need to specify: the training data, the model, a name for the model, and (optional) if your using wandb to log results

Training data - Options are '-a' for all data, '-tn' for news data, 'tw' for wikipedia data, and 'ti' for wikinews data. You are able to chain -tn, -tw, and -ti. You give it a 1 for that option.

The model - Options are '-rf' for random forest, '-ab' for adaboost, and '-cm' for adaboost and randomforest combined. You give it a 1 for that option.

The name of the model - Denoted by '-mn', you type out what you want to call the model.

Output - A pickled model

Example: python3 train_model -tn 1 -ab 1 -mn news_ada
Output: models/news_ada.sav

### 3. To use a model (run_model.py), these are the following instructions

You need to specify: the testing data, the name of the model your using and optionally you can turn off the options for evaluation and prediction if your using the shared task data.

Testing data - Options are '-n' for news data, '-w' for wikipedia data, '-i' for wikinews data. These options are followed by a 1 for that option.

OR - If you are using your own test data, you need to do '-t' followed by the name of the file. Such as '-t data_file'. (Don't include .csv)

Name of model - From your models folder, type out the name of the model you want to use (not including .sav), using '-mn'

Optionally - Predictions and evaluation scripts are ran automatically for shared task data. Though you may turned these off with a 0. The options are '-p' and '-e'.

Output - A csv file contaning all the outputs and features in results folder

Example: python3 run_model -t my_data_file -mn news_ada
Output - results/my_data_file_model_name\_\_results.csv

## Features

- camb_model - Model from https://github.com/siangooding/cwi_2018 (Gooding and Kochmar 2018, 'CAMB at CWI Shared Task 2018: Complex Word Identification with Ensemble-Based Voting', https://www.aclweb.org/anthology/W18-0520.pdf)

- testing_data - Process and format data for input to CAMB model

### Maintainers

Hansel Guzman-Soto (soto26938@gmail.com)\
Carson Vandegriffe (vandegc2@wwu.edu)

### Alumni

Hannah Buzard (buzardh@wwu.edu)

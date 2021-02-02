import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_html_components as html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import flask
import plotly.figure_factory as ff


def main(df):
    correct_df = get_correct_words(name)
    wrong_df = get_wrong_words(name)
    # fig = px.scatter(df, x="word", y="output")
    # rising_histo(df)
    # rising_histo(correct_df)
    # rising_histo(wrong_df)
    # scatter(df)
    # all_plots(df)
    all_plots(correct_df)
    all_plots(wrong_df)
##########################################################################################################

# given name of dataframe, returns all correct information


def get_correct_words(name):
    df = pd.read_csv("results/" + name + ".csv")
    df = df[df['output'] == df['complex_binary']]
    return df
##########################################################################################################

# Given name of dataframe, returns all incorrect observations


def all_plots(df):
    fig = px.scatter_matrix(df,
                            dimensions=[
                                "syllables", "dep num", "length", "google frequency"],
                            color="complex_binary",
                            title="Scatter matrix of iris data set",
                            labels=["syllables", "dep", "length", "freq"])

    fig.update_traces(diagonal_visible=False)
    fig.show()


def scatter(df):

    fig = px.scatter(df, x="word", y="google frequency", color="output",
                     size='syllables', hover_data=['google frequency', 'length'])
    fig.show()


def get_wrong_words(name):
    df = pd.read_csv("results/" + name + ".csv")
    df = df[df['output'] != df['complex_binary']]
    return df

##########################################################################################################


def rising_histo(df):
    fig = px.histogram(df, x="word", y="output", color="complex_binary",
                       marginal="box", hover_data=df.columns)
    fig.show()


"""
Input: A dataframe containing a subset of a results csv file (words, outputs, labels, and features)
Output: A dataframe containing calculated statistical information 
"""


def get_statistics(df):
    pass


##########################################################################################################
"""
Input: A dataframe containing statistics 
Output: A bunch of to be determined plots
"""


def plot_statistics(df):
    sns.displot(df, x="length", hue="output", multiple="stack")


##########################################################################################################

"""
Generalize this code, so that it takes in arguements where the user manually inputs the datasets (actually that part is command)
And manual inputs for the features + type of plots
"""


if __name__ == "__main__":
    name = "wikipedia_test_all_data_ada_results"
    df = pd.read_csv("results/" + name + ".csv")
    main(df)

    # correct_df = get_correct_words("wikipedia_test_results")
    # incorrect_df = get_wrong_words("wikipedia_test_results")
    # plot_statistics(correct_df)
    # plot_statistics(incorrect_df)
    # plot_statistics(df)

    # plt.show()

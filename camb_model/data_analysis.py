
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import MinMaxScaler


def main(df):
    correct_df = get_correct_words(df)
    wrong_df = get_wrong_words(df)

    if(args.plot == 'dist'):
        dist_plot(df, "all_data", x_var, y_var)
        dist_plot(correct_df, "correct_outputs", x_var, y_var)
        dist_plot(wrong_df, "wrong_outputs", x_var, y_var)

    if (args.plot == 'scatter'):
        scatter_plot(df, "all_data", x_var, y_var)
        scatter_plot(correct_df, "correct_outputs", x_var, y_var)
        scatter_plot(wrong_df, "wrong_outputs", x_var, y_var)

    if (args.plot == 'box'):
        boxplot(df, "all_data", x_var, y_var)
        boxplot(correct_df, "correct_output", x_var, y_var)
        boxplot(wrong_df, "wrong_outputs", x_var, y_var)

    if (args.plot == "violin"):
        violin(df, "all_data", x_var, y_var)
        violin(correct_df, "correct_outputs", x_var, y_var)
        violin(wrong_df, "wrong_outputs", x_var, y_var)

    plt.show()


##########################################################################################################

# given name of dataframe, returns all correct outputs


def get_correct_words(df):
    correct_df = df.copy()
    correct_df = correct_df[correct_df['output']
                            == correct_df['complex_binary']]
    return df
##########################################################################################################

# given name of dataframe, returns all wrong outputs


def get_wrong_words(df):
    wrong_df = df.copy()
    wrong_df = wrong_df[wrong_df['output'] != wrong_df['complex_binary']]
    return wrong_df

##########################################################################################################


def dist_plot(df, name, x_var, y_var):
    # scale = MinMaxScaler().fit(df[[x_var]])
    # df[x_var] = scale.transform(df[[x_var]])

    sns.displot(df, x=x_var, hue=y_var, multiple="stack")
    # plt.ylabel('normalized count')
    plt.title(name)

##########################################################################################################


def scatter_plot(df, name, x_var, y_var):
    sns.scatterplot(data=df, x=x_var, y=y_var, hue="output")
    plt.title(name)
    plt.show()

##########################################################################################################


def boxplot(df, name, x_var, y_var):
    sns.boxplot(data=df, x=x_var,
                y=y_var, hue="output", palette="Set2")
    plt.title(name)
    plt.show()


##########################################################################################################


def violin(df, name, x_var, y_var):
    sns.violinplot(x=x_var, y=y_var, hue="output",
                   data=df, palette="muted", split=True)
    plt.title(name)
    plt.show()


##########################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Plot data')
    parser.add_argument('data', type=str, default=None)
    parser.add_argument('x_var',  type=str, default=None)
    parser.add_argument('y_var', type=str, default=None)
    parser.add_argument('plot',  type=str, default=None)
    args = parser.parse_args()
    return args


##########################################################################################################
"""
Generalize this code, so that it takes in arguements where the user manually inputs the datasets (actually that part is command)
And manual inputs for the features + type of plots
"""


if __name__ == "__main__":
    args = parse_args()

    data_name = args.data
    x_var = args.x_var
    y_var = args.y_var

    df = pd.read_csv("results/" + data_name + ".csv")
    main(df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import MinMaxScaler


def main(df_list):
    name = ["all_data", "correct_outputs", "wrong_outputs"]
    i = 0
    for df in df_list:

        if (args.plot == 'kde'):
            kde_plot(df, name[i], x_var)

        if(args.plot == 'hist'):
            hist_plot(df, name[i], x_var, y_var)

        if (args.plot == 'scatter'):
            scatter_plot(df, name[i], x_var, y_var)

        if (args.plot == 'box'):
            boxplot(df, name[i], x_var, y_var)

        if (args.plot == "violin"):
            violin(df, name[i], x_var, y_var)

        if (args.plot == "cluster"):
            cluster(df, name[i])

        i += 1

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


def kde_plot(df, name, x_var):
    # scale = MinMaxScaler().fit(df[[x_var]])
    # df[x_var] = scale.transform(df[[x_var]])
    sns.histplot(df, x=x_var, hue="complex_binary", kde=True, element="step")
    plt.title(name)
    plt.show()


def hist_plot(df, name, x_var, y_var):
    # scale = MinMaxScaler().fit(df[[x_var]])
    # df[x_var] = scale.transform(df[[x_var]])

    sns.histplot(df, x=x_var, hue="complex_binary", element="step")
    # plt.ylabel('normalized count')
    plt.title(name)
    plt.show()

##########################################################################################################


def scatter_plot(df, name, x_var, y_var):
    sns.scatterplot(data=df, x=x_var, y_var=y_var, hue="output")
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
def cluster(df, name):
    df_features = df.drop(
        ['pos', 'lemma', 'word', 'sentence', "ID", "clean sentence", "start_index", "end_index"], axis=1)
    sns.clustermap(df_features)
    plt.show()

##########################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Plot data')
    parser.add_argument('data', type=str, default=None)
    parser.add_argument('x_var',  type=str, default=None)
    parser.add_argument('y_var', type=str, default=None)
    parser.add_argument('plot', type=str, default=None)

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
    try:
        correct_df = get_correct_words(df)
        wrong_df = get_wrong_words(df)
        df_list = [df, correct_df, wrong_df]
    except:
        df_list = [df]

    main(df_list)

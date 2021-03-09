import pandas as pd
import numpy as np
import math

# df_char_bigram = pd.read_csv("google_char_bigrams.csv")
# df_char_trigram = pd.read_csv(
#     "google_char_trigrams.csv")

# total_frequency = sum(df_char_bigram['frequency'])
# df_char_bigram['probability'] = df_char_bigram['frequency']/total_frequency

# total_frequency = sum(df_char_trigram['frequency'])
# df_char_trigram['probability'] = df_char_trigram['frequency']/total_frequency


# df_char_bigram.to_csv('google_char_bigrams.csv', index=False)
# df_char_trigram.to_csv('google_char_trigrams.csv', index=False)


df_bigram = pd.read_csv("google_bigrams.csv", usecols=[
                        'word_1', 'word_2', 'frequency'], delim_whitespace=True)


total_freq = sum(df_bigram['frequency'])
df_bigram['probability'] = df_bigram['frequency']/total_freq


print(df_bigram)

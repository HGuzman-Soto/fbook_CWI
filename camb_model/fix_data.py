import pandas as pd
import numpy as np
import string

#df = pd.read_pickle('features/German_Train_allInfo')
#df = pd.read_pickle('features/German_Test_allInfo')

###########################################################################


# df = df.drop('parse', axis=1)

#cat feature encoding

# df['ner'] = df['ner'].astype('category')
# df['ner'] = df['ner'].cat.codes

# df['pos'] = df['pos'].astype('category')
# df['pos'] = df['pos'].cat.codes

# # gen column names for embedding split
# names = []

# for x in range(1, 101):
#     names.append("embed_" + str(x))
#     print(names[-1])


# df3 = df.wordvec.apply(pd.Series)
# df3.columns = [names]

# # remove NaNs -> 0
# df3.fillna(0, inplace=True)

# print(df3)
# print(df3.columns)

# df = pd.concat([df, df3], axis=1)

# #df.to_csv('outdf.csv', index=False)

#df.columns = [col[0] if type(col) == tuple else col for col in df.columns]




###########################################################################


#df.to_pickle('features/German_Train_MOD_allInfo')
#df.to_pickle('features/German_Test_MOD_allInfo')



############################################################################

location = "training_data/german/German_Train.tsv"
df = pd.read_table(location, names=('ID', 'sentence', 'start_index', 'end_index', 'word', 'total_native',
                                                    'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic'), encoding='utf-8-sig')

location = "training_data/german/German_Test.tsv"
df2 = pd.read_table(location, names=('ID', 'sentence', 'start_index', 'end_index', 'word', 'total_native',
                                                    'total_non_native', 'native_complex', 'non_native_complex', 'complex_binary', 'complex_probabilistic'), encoding='utf-8-sig')

df3 = pd.concat([df,df2], axis=0)

# We create a table that contains only the words
df3['split'] = df3['word'].apply(lambda x: x.split())
df3['count'] = df3['split'].apply(lambda x: len(x))
words = df3[df3['count'] == 1]

word_set = words.word.str.lower()
word_set = pd.DataFrame(word_set)
word_set.columns = ['word']


##########################################################################################################
print("begin cleaning")
# Cleaning function for words
remove = string.punctuation
remove = remove.replace("-", "")
remove = remove.replace("'", "")  # don't remove apostraphies

remove = remove + '“'
remove = remove + '”'

pattern = r"[{}]".format(remove)  # create the pattern
word_set['word'] = word_set['word'].apply(
    lambda x: x.translate({ord(char): None for char in remove}))

print("finish cleaning", "\n")

words = pd.Series(word_set['word'])
print(words)
print(df3)


# small up the 300d text data
# first break up the file using linux split command

# files = ["xaa", "xab", "xac", "xad", "xae"]
files = ["xae"]


for f in files:
    file = open(f, 'r')
    lines = file.readlines()

    file_out = open("smaller300dim.txt", 'a')

    for l in lines:
        word = l.split(" ")[0]

        #cleaning

        if "/" in word:
            word = word.split("/")[1]

        if ":" in word:
            word = word.split(":")[1]

        if(word in list(word_set['word'])):
            file_out.write(l)
            
    
    print(f," - done")
    file.close()


import pandas as pd
import re

def main():
    #prune_lines("Spanish")
    #get_strings()
    #for german, change latin regex to include german characters
    get_words()
    get_freq()


#removes all lines in the raw file that are not associated with the given language
def prune_lines(lang):
    with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/pruned-8.dat", "w", encoding="utf8") as output_file:
        with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/Lang-8.dat", "r", encoding="utf8") as raw_file:
            for line in raw_file:
                header = line[0:line.find(',[')]
                header = header.split(',')
                if lang in header[2]:
                    output_file.write(line)

def get_strings():
    with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/words-8.dat", "w", encoding="utf8") as output_file:
        with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/pruned-8.dat", "r", encoding="utf8") as pruned_file:
            for line in pruned_file:
                text = line[line.find(',[') + 2 : len(line) - 3]
                text = text.split("\",\"")
                for phrase in text:
                    output_file.write(phrase + "\n")

def get_words():
    with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/single-8.dat", "w", encoding="utf8") as output_file:
        with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/words-8.dat", "r", encoding="utf8") as phrase_file:
            for line in phrase_file:

                newline = ""

                line = line.lower()

                line = line.replace('[',' ')
                line = line.replace(']',' ')
                
                for char in line:
                    if char in 'abcdefghijklmnopqrstuvwxyzáéíóúüñ ':
                        newline += char

                #split by word
                newline = newline.split(' ')

                for word in newline:
                    if(len(word) != 0):
                        output_file.write(word + "\n")

def get_freq():
    with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/freq-8.csv", "w", encoding="utf8") as output_file:
        with open("C:/Users/rasmu/CWI/fbook_CWI/camb_model/one_off_scripts/testing/single-8.dat", "r", encoding="utf8") as word_file:
            frequency = {}
            for word in word_file:
                if word in frequency:
                    frequency[word] += 1
                else:
                    frequency[word] = 1
            
            for word in frequency:
                output_file.write(word[0:word.find('\n')] + "," + str(frequency[word])+'\n')

if __name__ == "__main__":
    main()
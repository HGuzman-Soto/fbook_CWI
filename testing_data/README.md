# Instructions

## 1. Run unpack_json.py to unpack json object into proper data format (text, id)

Note: you can either <br>
        1. manually place the json file in the directory and run unpack_json.py <br>
        2. or run unpack_json.py with --j command to automatically retrieve the last
        downloaded json file from the scraper


## 2. Next, run.py

This will run preprocessing on the text and extract content words using POS tagging and NER. 
It will also attach the starting and ending indexes of each content word



## Columns are identified as such:

* text - raw social media text. Includes post + comment
* id - postID_commentID_sentenceID 
* clean_text - preprocessed text
* content_word - extracted content word
* starting_index - beginning index of the content word, from the cleaned text
* ending_index - ending index of the content word, from the cleaned text

## Diagram of Logical Flow 
![GitHub Logo](diagrams/pipeline_high.png)


## Diagram of Pipeline Architecture 
![GitHub Logo](diagrams/pipeline_overview.png)




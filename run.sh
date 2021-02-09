#!/bin/env python

echo "running pipeline.."

python3 setup.py

cd camb_model

if [ ! -d "features" ] 
then
    unzip -d features features.zip 
fi

if [ ! -d "results" ]
then
    unzip -d results results.zip
fi


cd testing_data/

printf "\n\n"
read -p "Use YOUR downloaded FB data? (y) (n for arbitrary data):   " prescraped
if [ "$prescraped" == "n" ];
then
    testFile=$(ls json_files -t | head -1)
    mv "json_files/$testFile" "$testFile"
    python3 unpack_json.py
else
    python3 unpack_json.py --j 1
fi

printf '%.s-' {1..50}
printf "\nFirst 10 lines of the unpacked json file:\n\n"
head -10 temp_data.csv
printf "\n"
read -p "(Enter) Clean Text"

python3 run.py


newFile=$(ls data_files/data -t | head -1)

printf "\n\n"
printf '%.s-' {1..50}
printf "\nFirst 10 lines of the cleaned data file:\n\n"
head -10 "data_files/data/$newFile"
printf "\n"
read -p "(Enter) Feature Extraction"

cd ../..

if [ ! -d "stanford-corenlp-4.2.0" ]
then
    echo "This may take a while..."
    curl -O https://nlp.stanford.edu/software/stanford-corenlp-latest.zip
    unzip stanford-corenlp-latest.zip && rm stanford-corenlp-latest.zip
fi

cd stanford-corenlp-4.2.0
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000 &

cd ../camb_model

filename=$(echo "$newFile" | cut -f 1 -d '.')
python3 feature_extraction.py -t "$filename"

# printf "\nModel Training:\n"
# printf "Which dataset(s) would you like to train the model on? (wikipedia, wikinews, news)\n"
# read -p "Input datasets seperated by commas: " testDatasets
# printf "Which model type? (randomforest, adaboost, combined)\n"
# read -p "Input model type: " modelType
# read -p "Model Name: " modelName

# setArgs=$""
# case $testDatasets in

#     *"wikipedia"*)
#         setArgs=$"${setArgs} -tw 1"
#         ;;&

#     *"wikinews"*)
#         setArgs=$"${setArgs} -ti 1"
#         ;;&

#     *"news"*)
#         setArgs=$"${setArgs} -tn 1"
#         ;;&
#     *)
#         ;;
# esac
# case $modelType in

#     "randomforest")
#         modelArg=$"-rf 1"
#         ;;

#     "adaboost")
#         modelArg=$"-ab 1"
#         ;;
    
#     "combined")
#         modelArg=$"-cm 1"
#         ;;
    
#     *)
#         ;;
# esac

# nameArg=$"--model_name $modelName"

# cmd=$"python3 train_model.py $setArgs $modelArg $nameArg"
# eval $cmd

sleep 1
printf "\n\n"
printf '%.s-' {1..50}
printf "\nThe Trained Models are:"
ls -1a models | sed 's/\.[a-z]*//g'
printf "\n"
read -p "Which model to test on?:   " modelname

cmd=$"python3 run_model.py -t $filename -mn $modelname"
eval $cmd

results=$(ls results -t | head -1)
cat results/"$results" | column -t -s, | less -S
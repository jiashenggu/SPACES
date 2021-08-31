#! /bin/bash

#python extract_convert.py
#python extract_vectorize.py

for ((i=0; i<5; i++));
    do
        nohup python extract_model.py $i > ./log/sfzy_extract_model$i.log 2>&1 &
    done

#python seq2seq_convert.py
#python seq2seq_model.py
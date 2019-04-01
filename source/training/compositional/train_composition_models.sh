#!/bin/bash

declare -a algorithms=(add full_add matrix)
declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)

for algorithm in "${algorithms[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                allennlp train source/training/compositional/configurations/${algorithm}/${embeddings}_win${window}_${dim}d.json \
                -s output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/ \
                --include-package source &
            done
            wait
        done
    done
done



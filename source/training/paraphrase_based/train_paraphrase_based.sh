#!/bin/bash

declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)

for embeddings in "${emb_algorithms[@]}"
do
    for dim in "${dims[@]}"
    do
        for window in "${windows[@]}"
        do
            allennlp train source/training/paraphrase_based/configurations/${embeddings}_win${window}_${dim}d.json \
            -s output/paraphrase_based/${embeddings}/win${window}/${dim}d/ \
            --include-package source &
        done
        wait
    done
done



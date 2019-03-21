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
            python -m source.evaluation.distributional.top_k \
                    output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz \
                    ${dim} \
                    data/ncs_test.txt > output/distributional/${embeddings}/win${window}/${dim}d/test_top_k.txt &
        done
        wait
    done
done



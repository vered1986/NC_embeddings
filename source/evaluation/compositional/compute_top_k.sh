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
                python -m source.evaluation.compositional.top_k \
                        output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz \
                        ${dim} \
                        output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/test_vectors.npy \
                        data/ncs_test.txt > output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/test_top_k.txt &
            done
            wait
        done
    done
done



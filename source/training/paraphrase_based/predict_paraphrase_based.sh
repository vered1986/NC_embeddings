#!/bin/bash

declare -a algorithms=(cooccurrence backtranslation)
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
                python -m source.training.compositional.compute_vectors \
                            output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz \
                            data/ncs_test.txt \
                            output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/test_vectors ${dim} &
            done
            wait
        done
    done
done
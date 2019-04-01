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
            python -m source.training.compositional.compute_vectors \
                        output/paraphrase_based/${embeddings}/win${window}/${dim}d/model.tar.gz \
                        data/ncs_paraphrases_test.jsonl \
                        output/paraphrase_based/${embeddings}/win${window}/${dim}d/test_vectors &
        done
        wait
    done
done

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
                python -m source.training.paraphrase_based.compute_vectors \
                            output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz \
                            data/nc_vocab.txt \
                            output/word_vocab \
                            output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
            done
        done
        wait
    done
done

for algorithm in "${algorithms[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                gzip output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
            done
            wait
        done
    done
done


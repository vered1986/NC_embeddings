#!/bin/bash

declare -a dims=(100 200 300)
declare -a algorithms=(sg cbow)
declare -a windows=(2 5 10)

for window in "${windows[@]}"
do
    for algorithm in "${algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            mkdir -p output/distributional/word2vec\_${algorithm}/win${window}/${dim}d/;
            python -m source.training.distributional.word2vec.train_word2vec \
                output/en_corpus_bigrams \
                output/distributional/word2vec\_${algorithm}/win${window}/${dim}d/ \
                --window_size ${window} --embedding_dim ${dim} --algorithm ${algorithm} &
        done
        wait
    done
done


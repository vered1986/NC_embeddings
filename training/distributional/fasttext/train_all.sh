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
            mkdir -p output/distributional/fasttext\_${algorithm}/${dim}d/win${window}/;
            python -m training.distributional.fasttext.train_fasttext.py \
                output/en_corpus_ngrams_extended \
                output/distributional/fasttext\_${algorithm}/${dim}d/win${window}/ \
                --window_size ${window} --embedding_dim ${dim} --algorithm ${algorithm} &
        done
        wait
    done
done


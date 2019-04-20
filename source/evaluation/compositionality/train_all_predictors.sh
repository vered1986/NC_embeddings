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
            python -m source.evaluation.compositionality.predictor \
                    output/distributional/${embeddings}/win${window}/${dim}d/model.tar.gz \
                    source/evaluation/compositionality/data/reddy2011.csv > output/distributional/${embeddings}/win${window}/${dim}d/compositionality_results.txt &
        done
        wait
    done
done

declare -a algorithms=(compositional/add compositional/full_add compositional/matrix compositional/lstm)

for algorithm in "${algorithms[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                python -m source.evaluation.compositionality.predictor \
                    output/${algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz \
                    source/evaluation/compositionality/data/reddy2011.csv \
                    --is_compositional > output/${algorithm}/${embeddings}/win${window}/${dim}d/compositionality_results.txt &
            done
        done
        wait
    done
done

declare -a algorithms=(paraphrase_based/cooccurrence paraphrase_based/backtranslation)

for algorithm in "${algorithms[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                python -m source.evaluation.compositionality.predictor \
                    output/${algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz \
                    source/evaluation/compositionality/data/reddy2011.csv \
                    --is_paraphrase_based > output/${algorithm}/${embeddings}/win${window}/${dim}d/compositionality_results.txt &
            done
        done
    done
    wait
done
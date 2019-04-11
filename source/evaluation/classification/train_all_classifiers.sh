#!/bin/bash

declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)
declare -a datasets=(tratz_coarse_grained_lexical tratz_fine_grained_lexical tratz_coarse_grained_random tratz_fine_grained_random)

for dataset in "${datasets[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                mkdir -p output/distributional/${embeddings}/win${window}/${dim}d/classification/${dataset};
                python -m source.evaluation.classification.classifier \
                        output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz ${dim} \
                        source/evaluation/classification/data/${dataset} \
                        output/distributional/${embeddings}/win${window}/${dim}d/classification/${dataset} &
            done
        done
        wait
    done
done

declare -a algorithms=(compositional/add compositional/full_add compositional/matrix compositional/lstm paraphrase_based/cooccurrence paraphrase_based/backtranslation)

for dataset in "${datasets[@]}"
do
    for algorithm in "${algorithms[@]}"
    do
        for embeddings in "${emb_algorithms[@]}"
        do
            for dim in "${dims[@]}"
            do
                for window in "${windows[@]}"
                do
                    mkdir -p output/${algorithm}/${embeddings}/win${window}/${dim}d/classification/${dataset};
                    python -m source.evaluation.classification.classifier \
                            output/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt.gz ${dim} \
                            source/evaluation/classification/data/${dataset} \
                            output/${algorithm}/${embeddings}/win${window}/${dim}d/classification/${dataset} \
                            --is_compositional &
                done
            done
            wait
        done
    done
done


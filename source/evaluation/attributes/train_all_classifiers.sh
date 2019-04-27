#!/bin/bash

declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)
declare -a datasets=(a_weapon different_colours is_round made_of_metal)

for dataset in "${datasets[@]}"
do
    for embeddings in "${emb_algorithms[@]}"
    do
        for dim in "${dims[@]}"
        do
            for window in "${windows[@]}"
            do
                mkdir -p output/distributional/${embeddings}/win${window}/${dim}d/attributes/${dataset};
                python -m source.evaluation.attributes.classifier \
                        output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz ${dim} \
                        source/evaluation/attributes/data/${dataset} \
                        output/distributional/${embeddings}/win${window}/${dim}d/attributes/${dataset} &
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
                    mkdir -p output/${algorithm}/${embeddings}/win${window}/${dim}d/attributes/${dataset};
                    python -m source.evaluation.attributes.classifier \
                            output/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt.gz ${dim} \
                            source/evaluation/attributes/data/${dataset} \
                            output/${algorithm}/${embeddings}/win${window}/${dim}d/attributes/${dataset} \
                            --is_compositional &
                done
            done
            wait
        done
    done
done


#!/bin/bash

declare -a algorithms=(full_add matrix lstm)
declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)


for embeddings in "${emb_algorithms[@]}"
do
    for dim in "${dims[@]}"
    do
        for window in "${windows[@]}"
        do
            # Copy the distributional embeddings and add "dist_" to the beginning of each line
            gzip -cd output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz | sed 's/^/dist_/g' > output/compositional/add/${embeddings}/win${window}/${dim}d/embeddings.txt;

            for algorithm in "${algorithms[@]}"
            do
                cp output/compositional/add/${embeddings}/win${window}/${dim}d/embeddings.txt output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt;
            done
        done
    done
done

declare -a algorithms=(add full_add matrix lstm)

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
                            data/nc_vocab.txt \
                            output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
            done
            wait
        done
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
                gunzip output/compositional/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
            done
            wait
        done
    done
done


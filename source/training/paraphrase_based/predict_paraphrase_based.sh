#!/bin/bash

declare -a algorithms=(cooccurrence backtranslation)
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
            gzip -cd output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz | sed 's/^/dist_/g' > output/paraphrase_based/cooccurrence/${embeddings}/win${window}/${dim}d/embeddings.txt;
            cp output/paraphrase_based/cooccurrence/${embeddings}/win${window}/${dim}d/embeddings.txt output/paraphrase_based/backtranslation/${embeddings}/win${window}/${dim}d/embeddings.txt &
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
                python -m source.training.compositional.compute_vectors \
                            output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz \
                            data/nc_vocab.txt \
                            output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
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
                gzip output/paraphrase_based/${algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt &
            done
            wait
        done
    done
done


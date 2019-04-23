#!/bin/bash

declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)
declare -a comp_algorithms=(add full_add matrix lstm)
declare -a par_algorithms=(cooccurrence backtranslation)

for embeddings in "${emb_algorithms[@]}"
do
    for dim in "${dims[@]}"
    do
        for window in "${windows[@]}"
        do
            for comp_algorithm in "${comp_algorithms[@]}"
            do
                for par_algorithm in "${par_algorithms[@]}"
                do
                    out_path="output/combined_predictor/${embeddings}/win${window}/${dim}d/${comp_algorithm}/${par_algorithm}/compositionality_results.txt";
                    echo ${out_path}
                    mkdir -p ${out_path};

                    comp_emb="output/compositional/${comp_algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz";
                    par_emb="output/paraphrase_based/${par_algorithm}/${embeddings}/win${window}/${dim}d/model.tar.gz";

                    python -m   source.evaluation.compositionality.predictor_combination \
                                ${comp_emb}##${par_emb} \
                                source/evaluation/compositionality/data/reddy2011.csv > ${out_path} &
                done
            done
            wait
        done
    done
done
#!/bin/bash

declare -a emb_algorithms=(word2vec_sg word2vec_cbow glove fasttext_sg fasttext_cbow)
declare -a windows=(2 5 10)
declare -a dims=(100 200 300)
declare -a datasets=(tratz_coarse_grained_lexical tratz_fine_grained_lexical tratz_coarse_grained_random tratz_fine_grained_random)
declare -a comp_algorithms=(add full_add matrix lstm)
declare -a par_algorithms=(cooccurrence backtranslation)

for dataset in "${datasets[@]}"
do
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
                        out_path="output/combined_classification/${embeddings}/win${window}/${dim}d/${comp_algorithm}/${par_algorithm}/${dataset}";
                        echo ${out_path}
                        mkdir -p ${out_path};
                        
                        dist_emb="output/distributional/${embeddings}/win${window}/${dim}d/embeddings.txt.gz";
                        comp_emb="output/compositional/${comp_algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt.gz";
                        par_emb="output/paraphrase_based/${par_algorithm}/${embeddings}/win${window}/${dim}d/embeddings.txt.gz";
                        
                        python -m source.evaluation.classification.classifier_combination ${dist_emb}##${comp_emb}##${par_emb} source/evaluation/classification/data/${dataset}  ${out_path} &
                    done
                done
            done
        done
    done
    wait
done

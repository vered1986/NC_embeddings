#!/bin/bash

# Important notes!
# 1. Make sure you first download the GloVe implementation from https://github.com/stanfordnlp/GloVe, and have the
# files under the current directory in a subdirectory called GloVe.
# 2. Don't run compute_cooccurrence.sh in parallel for different window sizes, the temp files get mixed up.

declare -a dims=(50 100 200 300)
declare -a windows=(2 5 10)

# Download the GloVe code
git clone https://github.com/stanfordnlp/GloVe.git;
mv GloVe/* .;
rm GloVe;

# Compute co-occurrence matrix once for each window size
for window in "${windows[@]}"
do
    mkdir -p output/distributional/glove/cooc_win${window};
    mkdir -p output/distributional/glove/win${window};
    bash source/training/distributional/glove/compute_cooccurrence.sh \
            output/en_corpus_bigrams \
            output/distributional/glove/ ${window} &
done
wait

for window in "${windows[@]}"
do
    # Train for each dimension
    for dim in "${dims[@]}"
    do
        mkdir -p output/distributional/glove/${dim}d/win${window}/;
        bash source/training/distributional/glove/train.sh \
                output/en_corpus_bigrams \
                output/distributional/glove/ ${window} ${dim} &
    done
    wait

    # Gzip
    for dim in "${dims[@]}"
    do
        gzip output/distributional/glove/${dim}d/win${window}/embeddings.txt &
    done
done


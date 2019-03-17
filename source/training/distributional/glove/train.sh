#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

CORPUS=$1
VOCAB_FILE=${CORPUS}\_vocab.txt
OUT_DIR=$2
WINDOW_SIZE=$3
VECTOR_SIZE=$4
COOCCURRENCE_FILE=${OUT_DIR}/cooc_win${WINDOW_SIZE}/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=${OUT_DIR}/${VECTOR_SIZE}d/win${WINDOW_SIZE}/cooccurrence.shuf.bin
BUILDDIR=GloVe/build
SAVE_FILE=${OUT_DIR}/${VECTOR_SIZE}d/win${WINDOW_SIZE}/embeddings
VERBOSE=2
MEMORY=4.0
MAX_ITER=15
BINARY=2
NUM_THREADS=8
X_MAX=10

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

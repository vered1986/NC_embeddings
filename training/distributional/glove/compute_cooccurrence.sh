#!/bin/bash

make

CORPUS=$1
VOCAB_FILE=${CORPUS}\_vocab.txt
OUT_DIR=$2
WINDOW_SIZE=$3
COOCCURRENCE_FILE=${OUT_DIR}/cooc_win${WINDOW_SIZE}/cooccurrence.bin
VERBOSE=2
BUILDDIR=GloVe/build
MEMORY=4.0
VOCAB_MIN_COUNT=1

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE


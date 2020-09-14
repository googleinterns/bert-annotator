#!/bin/sh

# The unprocessed file has the format
# 1: {[parameters document 1]}
# 2: {[parameters document 2 ]}
# Here, we transform this into
# documents {[parameters document 1]}
# documents {[parameters document 2]}
###

mkdir -p data/input/preprocessed
corpus=$1
echo "Preprocessing file data/input/raw/"$corpus".textproto"

sed "s/^[[:digit:]]* : /documents /g" data/input/raw/$corpus.textproto >> \
    data/input/preprocessed/$corpus.textproto

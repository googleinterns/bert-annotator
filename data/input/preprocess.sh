#!/bin/sh

# The unprocessed file has the format
# 1: {[parameters document 1]}
# 2: {[parameters document 2 ]}
# Here, we transform this into
# documents {[parameters document 1]}
# documents {[parameters document 2]}
###

rm -rf input_data/preprocessed
mkdir input_data/preprocessed
corpus=$1
echo "Preprocessing file input_data/raw/"$corpus".textproto"

sed "s/^[[:digit:]]* : /documents /g" input_data/raw/$corpus.textproto >> \
    input_data/preprocessed/$corpus.textproto
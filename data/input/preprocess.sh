#!/bin/sh

#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the the Apache License v2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Shared constants and functions."""

import tensorflow_hub as hub
import tensorflow as tf

# HACK: Required to make bert.tokenization work with TF2.
tf.gfile = tf.io.gfile
from com_google_research_bert import tokenization  # pylint: disable=wrong-import-position

LABEL_CONTAINER_NAME = "lucid"

LF_ADDRESS_LABEL = "address"
LF_TELEPHONE_LABEL = "phone"

MAIN_LABEL_ADDRESS = "ADDRESS"
MAIN_LABEL_TELEPHONE = "TELEPHONE"
MAIN_LABELS = (MAIN_LABEL_ADDRESS, MAIN_LABEL_TELEPHONE)

# "O" is short for "outside" and a magic value used by seqeval
# Not assigning "O" the index 0 help to find bugs where the label is always set
# to zero.
LABEL_OUTSIDE = "O"
LABEL_BEGIN_TELEPHONE = "B-TELEPHONE"
LABEL_INSIDE_TELEPHONE = "I-TELEPHONE"
LABEL_BEGIN_ADDRESS = "B-ADDRESS"
LABEL_INSIDE_ADDRESS = "I-ADDRESS"
LABELS = (LABEL_BEGIN_TELEPHONE, LABEL_INSIDE_TELEPHONE, LABEL_OUTSIDE,
          LABEL_BEGIN_ADDRESS, LABEL_INSIDE_ADDRESS)

# Copied from tagging_data_lib.
UNK_TOKEN = "[UNK]"
PADDING_LABEL_ID = -1

MOVING_WINDOW_MASK_LABEL_ID = -2


def create_tokenizer_from_hub_module(module_url):
    """Get the vocab file and casing info from the Hub module."""
    model = hub.KerasLayer(module_url, trainable=False)
    vocab_file = model.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = model.resolved_object.do_lower_case.numpy()

    return tokenization.FullTokenizer(vocab_file=vocab_file,
                                      do_lower_case=do_lower_case)

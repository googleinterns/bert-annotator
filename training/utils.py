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

import collections
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

ADDITIONAL_LABELS = (
    "B-DATE", "I-DATE", "B-NUMBER", "I-NUMBER", "B-LETTERS_SEPARATE",
    "I-LETTERS_SEPARATE", "B-MEASURE", "I-MEASURE", "B-MONEY", "I-MONEY",
    "B-ELECTRONIC", "I-ELECTRONIC", "B-ROMAN_NUMERAL_AS_CARDINAL",
    "I-ROMAN_NUMERAL_AS_CARDINAL", "B-EMOTICON_EMOJI", "I-EMOTICON_EMOJI",
    "B-ABBREVIATION_TO_EXPAND", "I-ABBREVIATION_TO_EXPAND",
    "B-VERBATIM_SEQUENCE", "I-VERBATIM_SEQUENCE", "B-TIME", "I-TIME",
    "B-CONNECTOR_RANGE", "I-CONNECTOR_RANGE", "B-DURATION", "I-DURATION",
    "B-CONNECTOR_SILENT", "I-CONNECTOR_SILENT", "B-CONNECTOR_GENERAL",
    "I-CONNECTOR_GENERAL", "B-FRACTION", "I-FRACTION", "B-LETTERS_AS_WORD",
    "I-LETTERS_AS_WORD", "B-ORDINAL", "I-ORDINAL", "B-CONNECTOR_RATIO",
    "I-CONNECTOR_RATIO", "B-ROMAN_NUMERAL_AS_DEFINITE_ORDINAL",
    "I-ROMAN_NUMERAL_AS_DEFINITE_ORDINAL", "B-DIGITS", "I-DIGITS",
    "B-CONNECTOR_SCORE", "I-CONNECTOR_SCORE", "B-CHUNKED_NUMBER",
    "I-CHUNKED_NUMBER", "B-CONNECTOR_MATH", "I-CONNECTOR_MATH",
    "B-CONNECTOR_DIMENSION", "I-CONNECTOR_DIMENSION", "B-MULTI_UNIT_MEASURE",
    "I-MULTI_UNIT_MEASURE", "B-ROMAN_NUMERAL_AS_ORDINAL",
    "I-ROMAN_NUMERAL_AS_ORDINAL", "B-CHEMICAL_FORMULA", "I-CHEMICAL_FORMULA")

# Copied from tagging_data_lib.
UNK_TOKEN = "[UNK]"
PADDING_LABEL_ID = -1

MOVING_WINDOW_MASK_LABEL_ID = -2

BERT_SENTENCE_START = "[CLS]"
BERT_SENTENCE_SEPARATOR = "[SEP]"
BERT_SENTENCE_PADDING = "[PAD]"

LABEL_ID_MAP = {label: i for i, label in enumerate(LABELS + ADDITIONAL_LABELS)}

LabeledExample = collections.namedtuple(
    "LabeledExample",
    ["prefix", "selection", "suffix", "complete_text", "label"])


def create_tokenizer_from_hub_module(module_url):
    """Get the vocab file and casing info from the Hub module."""
    model = hub.KerasLayer(module_url, trainable=False)
    vocab_file = model.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = model.resolved_object.do_lower_case.numpy()

    return tokenization.FullTokenizer(vocab_file=vocab_file,
                                      do_lower_case=do_lower_case)


def split_into_words(text, tokenizer):
    """Splits the text given the tokenizer, but merges subwords."""
    words = tokenizer.tokenize(text)
    joined_words = []
    for word in words:
        if word.startswith("##"):
            joined_words[-1] += word[2:]
        else:
            joined_words.append(word)
    return joined_words


def remove_whitespace_and_parse(text, tokenizer):
    """Removes all whitespace and some special characters.

    The tokenizer discards some utf-8 characters, such as the right-to-left
    indicator. Applying the tokenizer is slow, but the safest way to guarantee
    consistent behaviour.
    """
    return "".join(split_into_words(text, tokenizer))

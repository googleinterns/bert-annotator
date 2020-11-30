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
from google.protobuf.internal.decoder import _DecodeVarint32

import protocol_buffer.document_pb2 as proto_document

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

_MAX_BINPROTO_PREFIX_LENGTH = 10

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


def _convert_token_boundaries_to_codeunits(document):
    """Converts the indicies of the token boundaries from codepoints to"
    codeunits.
    """

    text_as_string = document.text
    text_as_bytes = bytes(text_as_string, "utf-8")
    for token in document.token:
        prefix_as_bytes = text_as_bytes[:token.start]
        token_as_bytes = text_as_bytes[token.start:token.end + 1]
        prefix_as_string = prefix_as_bytes.decode("utf-8")
        token_as_string = token_as_bytes.decode("utf-8")
        token.start = len(prefix_as_string)
        token.end = len(prefix_as_string) + len(token_as_string) - 1
    return document


def get_labeled_text_from_linkfragment(path):
    """Provides an iterator over all labeled texts in the linkfragments."""
    with open(path, "r") as file:
        for linkfragment in file:
            text, label_description = linkfragment.split("\t")
            prefix, remaining_text = text.split("{{{")
            labeled_text, suffix = remaining_text.split("}}}")

            prefix = prefix.strip()
            labeled_text = labeled_text.strip()
            label_description = label_description.strip()
            suffix = suffix.strip()

            if label_description == LF_ADDRESS_LABEL:
                label = MAIN_LABEL_ADDRESS
            elif label_description == LF_TELEPHONE_LABEL:
                label = MAIN_LABEL_TELEPHONE
            else:
                label = LABEL_OUTSIDE

            text_without_braces = text.replace("{{{", "").replace("}}}", "")

            yield LabeledExample(prefix=prefix,
                                 selection=labeled_text,
                                 suffix=suffix,
                                 complete_text=text_without_braces,
                                 label=label)


def get_labeled_text_from_document(document, only_main_labels=False):
    """Provides an iterator over all labeled texts in the proto document."""
    text = document.text

    last_label_end = -1
    for label in document.labeled_spans[LABEL_CONTAINER_NAME].labeled_span:
        if only_main_labels and label.label not in MAIN_LABELS:
            continue

        label_start = document.token[label.token_start].start
        label_end = document.token[label.token_end].end

        prefix = text[last_label_end + 1:label_start]
        labeled_text = text[label_start:label_end + 1]

        last_label_end = label_end

        yield LabeledExample(prefix=prefix,
                             selection=labeled_text,
                             suffix="",
                             complete_text=None,
                             label=label.label)

    suffix = text[last_label_end + 1:]
    yield LabeledExample(prefix="",
                         selection="",
                         suffix=suffix,
                         complete_text=None,
                         label=LABEL_OUTSIDE)


def get_documents(path):
    """Provides an iterator over all documents, where the boundaries have been
    updated to use codeunits."""
    document = proto_document.Document()
    with open(path, "rb") as src_file:
        msg_buf = src_file.read(_MAX_BINPROTO_PREFIX_LENGTH)
        while msg_buf:
            # Get the message length.
            msg_len, new_pos = _DecodeVarint32(msg_buf, 1)
            msg_buf = msg_buf[new_pos:]
            # Read the rest of the message.
            msg_buf += src_file.read(msg_len - len(msg_buf))
            document.ParseFromString(msg_buf)
            msg_buf = msg_buf[msg_len:]
            # Read the length prefix for the next message.
            msg_buf += src_file.read(_MAX_BINPROTO_PREFIX_LENGTH)
            yield _convert_token_boundaries_to_codeunits(document)

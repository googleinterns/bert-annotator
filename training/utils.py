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
import os
from official.nlp.data import tagging_data_lib
import tensorflow_hub as hub
import tensorflow as tf
from absl import logging

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
    """Splits the text given the tokenizer."""
    return tokenizer.basic_tokenizer.tokenize(text)


def remove_whitespace_and_parse(text, tokenizer):
    """Removes all whitespace and some special characters.

    The tokenizer discards some utf-8 characters, such as the right-to-left
    indicator. Applying the tokenizer is slow, but the safest way to guarantee
    consistent behaviour.
    """
    return "".join(split_into_words(text, tokenizer))


def add_tfrecord_label(text, label, tokenizer, example, use_additional_labels):
    """Adds one label for each word in the text to the example."""
    label_id_map = {label: i for i, label in enumerate(LABELS)}

    words = split_into_words(text, tokenizer)
    if label in MAIN_LABELS or (use_additional_labels
                                and label in MAIN_LABELS + ADDITIONAL_LABELS):
        example.add_word_and_label_id(words[0], label_id_map["B-%s" % label])
        for word in words[1:]:
            example.add_word_and_label_id(word, label_id_map["I-%s" % label])
    else:
        for word in words:
            example.add_word_and_label_id(word, label_id_map[LABEL_OUTSIDE])


def _tokenize_example(example,
                      max_length,
                      tokenizer,
                      text_preprocessing=None,
                      moving_window_overlap=20,
                      mask_overlap=False):
    """Tokenizes words and breaks long example into short ones.

    Very similiar to _tokenize_example in tagging_data_lib, but implements a
    moving window. The tokens closest to the border are repeated in the next
    sub-sentence. The half of the repeated tokens that are closest to the border
    are not labeled if mask_overlap is True.
    """
    assert moving_window_overlap % 2 == 0, "moving_window_overlap must be even."
    half_moving_window_overlap = moving_window_overlap // 2
    moving_window_padding = [MOVING_WINDOW_MASK_LABEL_ID
                             ] * half_moving_window_overlap
    # Needs additional [CLS] and [SEP] tokens and space for the moving window.
    max_length = max_length - 2 - moving_window_overlap
    new_examples = []
    new_example = tagging_data_lib.InputExample(
        sentence_id=example.sentence_id, sub_sentence_id=0)
    for i, word in enumerate(example.words):
        if any([x < 0 for x in example.label_ids]):
            raise ValueError("Unexpected negative label_id: %s" %
                             example.label_ids)

        if text_preprocessing:
            word = text_preprocessing(word)
        subwords = tokenizer.tokenize(word)
        if (not subwords or len(subwords) > max_length) and word:
            subwords = [UNK_TOKEN]

        if len(subwords) + len(new_example.words) > max_length:
            # A copy is needed as the original list is modified below.
            previous_label_ids = new_example.label_ids.copy()
            previous_label_words = new_example.words

            if mask_overlap and moving_window_overlap > 0:
                # The last tokens have very little context, they are labeled in
                # the next sub-sentence.
                new_example.label_ids[
                    -half_moving_window_overlap:] = moving_window_padding

            # Start a new example.
            new_examples.append(new_example)
            last_sub_sentence_id = new_example.sub_sentence_id
            new_example = tagging_data_lib.InputExample(
                sentence_id=example.sentence_id,
                sub_sentence_id=last_sub_sentence_id + 1)

            if moving_window_overlap > 0:
                # The previously masked tokens need to be labeled, additional
                # tokens are copied and masked to be used as context.
                new_example.words.extend(
                    previous_label_words[-moving_window_overlap:])
                if mask_overlap:
                    new_example.label_ids.extend(moving_window_padding)
                    new_example.label_ids.extend(
                        previous_label_ids[-half_moving_window_overlap:])
                else:
                    new_example.label_ids.extend(
                        previous_label_ids[-moving_window_overlap:])

        for j, subword in enumerate(subwords):
            # Use the real label for the first subword, and pad label for
            # the remainings.
            subword_label = example.label_ids[i] if j == 0 else PADDING_LABEL_ID
            new_example.add_word_and_label_id(subword, subword_label)

    assert new_example.words
    new_examples.append(new_example)

    return new_examples


def write_example_to_file(examples,
                          tokenizer,
                          max_seq_length,
                          output_file,
                          text_preprocessing=None,
                          moving_window_overlap=20,
                          mask_overlap=False):
    """Writes `InputExample`s to a tfrecord file with `tf.train.Example` protos.

    Identical to tagging_data_lib.write_example_to_file except for the
    additional parameters that are passed to _tokenize_example.


    Args:
        examples: A list of `InputExample` instances.
        tokenizer: The tokenizer to be applied on the data.
        max_seq_length: Maximum length of generated sequences.
        output_file: The name of the output tfrecord file.
        text_preprocessing: optional preprocessing run on each word prior to
            tokenization.
        moving_window_overlap: Size of the moving window.
        mask_overlap: Whether to mask the overlap introduced by the moving
            window or not.

    Returns:
        The total number of tf.train.Example proto written to file.
    """
    tf.io.gfile.makedirs(os.path.dirname(output_file))
    writer = tf.io.TFRecordWriter(output_file)
    num_tokenized_examples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d to %s", ex_index,
                         len(examples), output_file)

        tokenized_examples = _tokenize_example(example, max_seq_length,
                                               tokenizer, text_preprocessing,
                                               moving_window_overlap,
                                               mask_overlap)
        num_tokenized_examples += len(tokenized_examples)
        for per_tokenized_example in tokenized_examples:
            # pylint: disable=protected-access
            tf_example = tagging_data_lib._convert_single_example(
                per_tokenized_example, max_seq_length, tokenizer)
            # pylint: enable=protected-access
            writer.write(tf_example.SerializeToString())

    writer.close()
    return num_tokenized_examples

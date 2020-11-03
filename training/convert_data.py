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
"""Converts the given data from .binproto or .lftxt format to .tfrecord"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app, flags
import tensorflow_hub as hub
import tensorflow as tf
from official.nlp.data import tagging_data_lib

import protocol_buffer.documents_pb2 as proto_documents
from training.utils import (LABELS, LABEL_CONTAINER_NAME, MAIN_LABELS,
                            MAIN_LABEL_ADDRESS, MAIN_LABEL_TELEPHONE,
                            LABEL_OUTSIDE)

tf.gfile = tf.io.gfile  # HACK: Required to make bert.tokenization work with TF2
from com_google_research_bert import tokenization  # pylint: disable=wrong-import-position

FLAGS = flags.FLAGS
LF_ADDRESS_LABEL = "address"
LF_TELEPHONE_LABEL = "phone"

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximal sequence length. Longer sequences are split.")
flags.DEFINE_string(
    "train_data_input_path", None,
    "The path to the (augmented) training data in .binproto/.tftxt format.")
flags.DEFINE_string(
    "eval_data_input_path", None,
    "The path to the (augmented) evaluation data in .binproto/.tftxt format.")
flags.DEFINE_multi_string(
    "test_data_input_path", [],
    "The path to the test data in .binproto/.tftxt format. May be defined more"
    " than once.")
flags.DEFINE_string(
    "train_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records.")
flags.DEFINE_string(
    "eval_data_output_path", None,
    "The path in which generated evaluation input data will be written as tf"
    " records.")
flags.DEFINE_multi_string(
    "test_data_output_path", [],
    "The path in which generated test input data will be written as tf records."
    " May be defined more than once, in the same order as"
    " test_data_input_path.")
flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")


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
        token.end = len(prefix_as_string) + len(token_as_string)
    return document


def _add_label(text, label, tokenizer, example):
    label_id_map = {label: i for i, label in enumerate(LABELS)}

    words = _split_into_words(text, tokenizer)
    if label in MAIN_LABELS:
        example.add_word_and_label_id(words[0], label_id_map["B-%s" % label])
        for word in words[1:]:
            example.add_word_and_label_id(word, label_id_map["I-%s" % label])
    else:
        for word in words:
            example.add_word_and_label_id(word, label_id_map[LABEL_OUTSIDE])


def _read_binproto(file_name, tokenizer):
    """Reads one file and returns a list of `InputExample` instances."""
    documents = proto_documents.Documents()
    with open(file_name, "rb") as src_file:
        documents.ParseFromString(src_file.read())

    examples = []
    sentence_id = 0
    example = tagging_data_lib.InputExample(sentence_id=0)
    for document in documents.documents:
        document = _convert_token_boundaries_to_codeunits(document)
        text = document.text

        last_label_end = -1
        for label in document.labeled_spans[LABEL_CONTAINER_NAME].labeled_span:
            label_start = document.token[label.token_start].start
            label_end = document.token[label.token_end].end

            _add_label(text[last_label_end + 1:label_start], LABEL_OUTSIDE,
                       tokenizer, example)
            _add_label(text[label_start:label_end + 1], label.label, tokenizer,
                       example)

            last_label_end = label_end
        _add_label(text[last_label_end + 1:], LABEL_OUTSIDE, tokenizer,
                   example)

        if example.words:
            examples.append(example)
            sentence_id += 1
            example = tagging_data_lib.InputExample(sentence_id=sentence_id)
    return examples


def _read_lftxt(file_name, tokenizer):
    """Reads one file and returns a list of `InputExample` instances."""
    examples = []
    sentence_id = 0
    example = tagging_data_lib.InputExample(sentence_id=0)
    with open(file_name, "r") as src_file:
        for line in src_file:
            text, label_description = line.split("\t")
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

            _add_label(prefix, LABEL_OUTSIDE, tokenizer, example)
            _add_label(labeled_text, label, tokenizer, example)
            _add_label(suffix, LABEL_OUTSIDE, tokenizer, example)

            if example.words:
                examples.append(example)
                sentence_id += 1
                example = tagging_data_lib.InputExample(
                    sentence_id=sentence_id)
    return examples


def _generate_tf_records(tokenizer, max_seq_length, train_examples,
                         eval_examples, test_input_data_examples,
                         train_data_output_path, eval_data_output_path):
    """Generates tfrecord files from the `InputExample` lists."""
    common_kwargs = dict(tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         text_preprocessing=tokenization.convert_to_unicode)

    train_data_size = tagging_data_lib.write_example_to_file(
        train_examples, output_file=train_data_output_path, **common_kwargs)

    eval_data_size = tagging_data_lib.write_example_to_file(
        eval_examples, output_file=eval_data_output_path, **common_kwargs)

    test_data_size = {}
    for output_path, examples in test_input_data_examples.items():
        test_data_size[output_path] = tagging_data_lib.write_example_to_file(
            examples, output_file=output_path, **common_kwargs)

    meta_data = tagging_data_lib.token_classification_meta_data(
        train_data_size,
        max_seq_length,
        len(LABELS),
        eval_data_size,
        test_data_size,
        label_list=LABELS,
        processor_type="text_classifier")
    return meta_data


def _create_tokenizer_from_hub_module(module_url):
    """Get the vocab file and casing info from the Hub module."""
    model = hub.KerasLayer(module_url, trainable=False)
    vocab_file = model.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = model.resolved_object.do_lower_case.numpy()

    return tokenization.FullTokenizer(vocab_file=vocab_file,
                                      do_lower_case=do_lower_case)


def _split_into_words(text, tokenizer):
    """Splits the text given the tokenizer, but merges subwords."""
    words = tokenizer.tokenize(text)
    joined_words = []
    for word in words:
        if word.startswith("##"):
            joined_words[-1] += word[2:]
        else:
            joined_words.append(word)
    return joined_words


def main(_):
    if len(FLAGS.test_data_input_path) != len(FLAGS.test_data_output_path):
        raise ValueError("Specify an output path for each test input")

    tokenizer = _create_tokenizer_from_hub_module(FLAGS.module_url)

    if FLAGS.train_data_input_path.endswith(".binproto"):
        train_examples = _read_binproto(FLAGS.train_data_input_path, tokenizer)
    elif FLAGS.train_data_input_path.endswith(".lftxt"):
        train_examples = _read_lftxt(FLAGS.train_data_input_path, tokenizer)
    else:
        raise ValueError(
            "Invalid file format, only .binproto and .lftxt are supported.")

    if FLAGS.eval_data_input_path.endswith(".binproto"):
        eval_examples = _read_binproto(FLAGS.eval_data_input_path, tokenizer)
    elif FLAGS.eval_data_input_path.endswith(".lftxt"):
        eval_examples = _read_lftxt(FLAGS.eval_data_input_path, tokenizer)
    else:
        raise ValueError(
            "Invalid file format, only .binproto and .lftxt are supported.")

    test_examples = {}
    for input_path, output_path in zip(FLAGS.test_data_input_path,
                                       FLAGS.test_data_output_path):
        print("Paths: ", input_path, output_path)
        if input_path.endswith(".binproto"):
            test_examples[output_path] = _read_binproto(input_path, tokenizer)
        elif input_path.endswith(".lftxt"):
            test_examples[output_path] = _read_lftxt(input_path, tokenizer)
        else:
            raise ValueError(
                "Invalid file format, only .binproto and .lftxt are supported."
            )

    meta_data = _generate_tf_records(tokenizer, FLAGS.max_seq_length,
                                     train_examples, eval_examples,
                                     test_examples,
                                     FLAGS.train_data_output_path,
                                     FLAGS.eval_data_output_path)
    tf.io.gfile.makedirs(os.path.dirname(FLAGS.meta_data_file_path))
    with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
        writer.write(json.dumps(meta_data, indent=4) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("train_data_input_path")  #
    flags.mark_flag_as_required("eval_data_input_path")
    flags.mark_flag_as_required("train_data_output_path")
    flags.mark_flag_as_required("eval_data_output_path")
    flags.mark_flag_as_required("meta_data_file_path")

    app.run(main)

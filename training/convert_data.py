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
import tensorflow as tf
from official.nlp.data import tagging_data_lib
from com_google_research_bert import tokenization

from training.utils import (LABELS, MAIN_LABELS, LABEL_OUTSIDE,
                            ADDITIONAL_LABELS,
                            create_tokenizer_from_hub_module, split_into_words,
                            write_example_to_file)
from training.file_reader import get_file_reader

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximal sequence length. Longer sequences are split.")
flags.DEFINE_string(
    "train_data_input_path", None,
    "The path to the (augmented) training data in .binproto/.tftxt format.")
flags.DEFINE_string(
    "dev_data_input_path", None,
    "The path to the (augmented) development data in .binproto/.tftxt format.")
flags.DEFINE_multi_string(
    "test_data_input_paths", [],
    "The paths to the test data in .binproto/.tftxt format. May be defined more"
    " than once.")
flags.DEFINE_string(
    "train_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records.")
flags.DEFINE_string(
    "dev_data_output_path", None,
    "The path in which generated development input data will be written as tf"
    " records.")
flags.DEFINE_multi_string(
    "test_data_output_paths", [],
    "The paths in which generated test input data will be written as tf"
    " records. May be defined more than once, in the same order as"
    " test_data_input_paths.")
flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")
flags.DEFINE_integer(
    "moving_window_overlap", 20, "The size of the overlap for a moving window."
    " Setting it to zero restores the default behaviour of hard splitting.")
flags.DEFINE_boolean(
    "train_with_additional_labels", False,
    "If set, the flags other than address/phone are used for training, too.")

FLAGS = flags.FLAGS


def _add_label(text, label, tokenizer, example, use_additional_labels):
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


def _generate_tf_records(tokenizer, max_seq_length, train_examples,
                         dev_examples, test_input_data_examples,
                         train_data_output_path, dev_data_output_path,
                         meta_data_file_path, moving_window_overlap):
    """Generates tfrecord files from the `InputExample` lists."""
    common_kwargs = dict(tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         text_preprocessing=tokenization.convert_to_unicode)

    train_data_size = write_example_to_file(
        train_examples,
        output_file=train_data_output_path,
        **common_kwargs,
        moving_window_overlap=moving_window_overlap,
        mask_overlap=False)

    dev_data_size = write_example_to_file(
        dev_examples,
        output_file=dev_data_output_path,
        **common_kwargs,
        moving_window_overlap=moving_window_overlap,
        mask_overlap=True)

    test_data_size = {}
    for output_path, examples in test_input_data_examples.items():
        test_data_size[output_path] = write_example_to_file(
            examples,
            output_file=output_path,
            **common_kwargs,
            moving_window_overlap=moving_window_overlap,
            mask_overlap=True)

    meta_data = tagging_data_lib.token_classification_meta_data(
        train_data_size,
        max_seq_length,
        len(LABELS),
        dev_data_size,
        test_data_size,
        label_list=LABELS,
        processor_type="text_classifier")

    tf.io.gfile.makedirs(os.path.dirname(meta_data_file_path))
    with tf.io.gfile.GFile(meta_data_file_path, "w") as writer:
        writer.write(json.dumps(meta_data, indent=4) + "\n")


def main(_):
    if len(FLAGS.test_data_input_paths) != len(FLAGS.test_data_output_paths):
        raise ValueError("Specify an output path for each test input")

    tokenizer = create_tokenizer_from_hub_module(FLAGS.module_url)

    train_examples = get_file_reader(FLAGS.train_data_input_path).get_examples(
        tokenizer,
        FLAGS.train_with_additional_labels,
        use_gold_tokenization_and_include_target_labels=True)
    dev_examples = get_file_reader(FLAGS.dev_data_input_path).get_examples(
        tokenizer,
        FLAGS.train_with_additional_labels,
        use_gold_tokenization_and_include_target_labels=True)
    test_examples = {}
    for input_path, output_path in zip(FLAGS.test_data_input_paths,
                                       FLAGS.test_data_output_paths):
        test_examples[output_path] = get_file_reader(input_path).get_examples(
            tokenizer,
            use_additional_labels=False,
            use_gold_tokenization_and_include_target_labels=False)

    _generate_tf_records(tokenizer, FLAGS.max_seq_length, train_examples,
                         dev_examples, test_examples,
                         FLAGS.train_data_output_path,
                         FLAGS.dev_data_output_path, FLAGS.meta_data_file_path,
                         FLAGS.moving_window_overlap)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("train_data_input_path")
    flags.mark_flag_as_required("dev_data_input_path")
    flags.mark_flag_as_required("train_data_output_path")
    flags.mark_flag_as_required("dev_data_output_path")
    flags.mark_flag_as_required("meta_data_file_path")

    app.run(main)

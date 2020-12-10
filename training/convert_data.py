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

from absl import app, flags, logging
import tensorflow as tf
from official.nlp.data import tagging_data_lib
from com_google_research_bert import tokenization

from training.utils import (LABELS, MAIN_LABELS, LABEL_OUTSIDE, UNK_TOKEN,
                            PADDING_LABEL_ID, MOVING_WINDOW_MASK_LABEL_ID,
                            ADDITIONAL_LABELS,
                            create_tokenizer_from_hub_module, split_into_words,
                            get_labeled_text_from_linkfragment, get_documents,
                            get_labeled_text_from_document)

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
    # Needs additional [CLS] and [SEP] tokens.
    max_length = max_length - 2
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


def _write_example_to_file(examples,
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


def _read_binproto(file_name, tokenizer, use_additional_labels,
                   use_gold_tokenization_and_include_target_labels):
    """Reads one file and returns a list of `InputExample` instances."""
    examples = []
    sentence_id = 0
    example = tagging_data_lib.InputExample(sentence_id=0)
    for document in get_documents(file_name):
        text = document.text

        if use_gold_tokenization_and_include_target_labels:
            for labeled_example in get_labeled_text_from_document(document):
                _add_label(labeled_example.prefix, LABEL_OUTSIDE, tokenizer,
                           example, use_additional_labels)
                _add_label(labeled_example.selection, labeled_example.label,
                           tokenizer, example, use_additional_labels)
        else:
            # The tokenizer will split the text without taking the target labels
            # into account.
            _add_label(text, LABEL_OUTSIDE, tokenizer, example,
                       use_additional_labels)

        if example.words:
            examples.append(example)
            sentence_id += 1
            example = tagging_data_lib.InputExample(sentence_id=sentence_id)
    return examples


def _read_lftxt(path, tokenizer, use_additional_labels,
                use_gold_tokenization_and_include_target_labels):
    """Reads one file and returns a list of `InputExample` instances."""
    examples = []
    sentence_id = 0
    example = tagging_data_lib.InputExample(sentence_id=0)
    prev_text = ""
    for labeled_example in get_labeled_text_from_linkfragment(path):

        if use_gold_tokenization_and_include_target_labels:
            if prev_text == labeled_example.complete_text:
                raise NotImplementedError(
                    "Merging non-overlapping labels from multiple instances of"
                    " the same sentence is not implemented. Sentences can only"
                    " be merged if no corresponding labels are saved.")
            _add_label(labeled_example.prefix, LABEL_OUTSIDE, tokenizer,
                       example, use_additional_labels)
            _add_label(labeled_example.selection, labeled_example.label,
                       tokenizer, example, use_additional_labels)
            _add_label(labeled_example.suffix, LABEL_OUTSIDE, tokenizer,
                       example, use_additional_labels)
        else:
            if prev_text == labeled_example.complete_text:
                continue
            _add_label(labeled_example.complete_text,
                       LABEL_OUTSIDE,
                       tokenizer,
                       example,
                       use_additional_labels=False)
        prev_text = labeled_example.complete_text

        if example.words:
            examples.append(example)
            sentence_id += 1
            example = tagging_data_lib.InputExample(sentence_id=sentence_id)
    return examples


def _generate_tf_records(tokenizer, max_seq_length, train_examples,
                         dev_examples, test_input_data_examples,
                         train_data_output_path, dev_data_output_path,
                         meta_data_file_path, moving_window_overlap):
    """Generates tfrecord files from the `InputExample` lists."""
    common_kwargs = dict(tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         text_preprocessing=tokenization.convert_to_unicode)

    train_data_size = _write_example_to_file(
        train_examples,
        output_file=train_data_output_path,
        **common_kwargs,
        moving_window_overlap=moving_window_overlap,
        mask_overlap=False)

    dev_data_size = _write_example_to_file(
        dev_examples,
        output_file=dev_data_output_path,
        **common_kwargs,
        moving_window_overlap=moving_window_overlap,
        mask_overlap=True)

    test_data_size = {}
    for output_path, examples in test_input_data_examples.items():
        test_data_size[output_path] = _write_example_to_file(
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


def _get_examples(path, tokenizer, train_with_additional_labels,
                  use_gold_tokenization_and_include_target_labels):
    """Loads the data from a .binproto or .lftxt file."""
    if path.endswith(".binproto"):
        examples = _read_binproto(
            path, tokenizer, train_with_additional_labels,
            use_gold_tokenization_and_include_target_labels)
    elif path.endswith(".lftxt"):
        examples = _read_lftxt(
            path, tokenizer, train_with_additional_labels,
            use_gold_tokenization_and_include_target_labels)
    else:
        raise ValueError(
            "Invalid file format, only .binproto and .lftxt are supported.")
    return examples


def main(_):
    if len(FLAGS.test_data_input_paths) != len(FLAGS.test_data_output_paths):
        raise ValueError("Specify an output path for each test input")

    tokenizer = create_tokenizer_from_hub_module(FLAGS.module_url)

    train_examples = _get_examples(
        FLAGS.train_data_input_path,
        tokenizer,
        FLAGS.train_with_additional_labels,
        use_gold_tokenization_and_include_target_labels=True)
    dev_examples = _get_examples(
        FLAGS.dev_data_input_path,
        tokenizer,
        FLAGS.train_with_additional_labels,
        use_gold_tokenization_and_include_target_labels=True)
    test_examples = {}
    for input_path, output_path in zip(FLAGS.test_data_input_paths,
                                       FLAGS.test_data_output_paths):
        test_examples[output_path] = _get_examples(
            input_path,
            tokenizer,
            train_with_additional_labels=False,
            use_gold_tokenization_and_include_target_labels=False)

    moving_window_overlap = FLAGS.moving_window_overlap
    if moving_window_overlap % 2 != 0:
        raise ValueError("moving_window_overlap must be even.")

    _generate_tf_records(tokenizer, FLAGS.max_seq_length, train_examples,
                         dev_examples, test_examples,
                         FLAGS.train_data_output_path,
                         FLAGS.dev_data_output_path, FLAGS.meta_data_file_path,
                         moving_window_overlap)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("train_data_input_path")
    flags.mark_flag_as_required("dev_data_input_path")
    flags.mark_flag_as_required("train_data_output_path")
    flags.mark_flag_as_required("dev_data_output_path")
    flags.mark_flag_as_required("meta_data_file_path")

    app.run(main)

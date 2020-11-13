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
"""Evaluates the trained model on the given test set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app, flags
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
import tensorflow as tf

from official.nlp.tasks.tagging import TaggingConfig, TaggingTask, predict
from official.nlp.data import tagging_dataloader
from training.utils import (BERT_SENTENCE_PADDING, BERT_SENTENCE_SEPARATOR,
                            BERT_SENTENCE_START, LABELS, MAIN_LABELS,
                            PADDING_LABEL_ID, MOVING_WINDOW_MASK_LABEL_ID,
                            create_tokenizer_from_hub_module)

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_string("model_path", None, "The path to the trained model.")
flags.DEFINE_multi_string("test_data_paths", [],
                          "The path to the test data in .tfrecord format.")
flags.DEFINE_string(
    "visualisation_folder", None,
    "If set, a comparison of the target/hypothesis labeling is saved in .html"
    " format")
flags.DEFINE_boolean(
    "strict_eval", False,
    "Only used for scoring. If True, a label must not begin with an 'I-' tag.")

FLAGS = flags.FLAGS


def _infer(module_url, model_path, test_data_path):
    """Computes the predicted label sequence using the trained model."""
    test_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=test_data_path,
        seq_length=128,
        global_batch_size=64,
        is_training=False,
        include_sentence_id=True,
        drop_remainder=False)
    config = TaggingConfig(hub_module_url=module_url, class_names=LABELS)
    task = TaggingTask(config)
    model = task.build_model()
    model.load_weights(model_path)
    predictions = predict(task, test_data_config, model)

    merged_predictions = []
    for _, part_id, predicted_labels in predictions:
        if part_id == 0:
            merged_predictions.append(predicted_labels)
        else:
            merged_predictions[-1].extend(predicted_labels)

    return merged_predictions


def _extract_target_labels(module_url, trg_path):
    """Extracts the target labels from the given .tfrecord file."""
    feature_description = {
        "input_ids":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "input_mask":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "label_ids":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sentence_id":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sub_sentence_id":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    tokenizer = create_tokenizer_from_hub_module(module_url)
    targets = []
    texts = []
    raw_dataset = tf.data.TFRecordDataset(trg_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    target_labels = []
    tokens = []
    for features in parsed_dataset:
        if features["sub_sentence_id"][0] == 0:
            if target_labels:
                targets.append(target_labels)
                texts.append(tokens)
            target_labels = []
            tokens = []

        for token_id, label_id in zip(features["input_ids"].numpy(),
                                      features["label_ids"].numpy()):
            token = tokenizer.convert_ids_to_tokens([token_id
                                                     ])[0].replace("##", "")
            if token in [
                    BERT_SENTENCE_START, BERT_SENTENCE_SEPARATOR,
                    BERT_SENTENCE_PADDING
            ]:
                continue
            if label_id == MOVING_WINDOW_MASK_LABEL_ID:
                continue
            elif label_id == PADDING_LABEL_ID:
                tokens[-1] += token
            else:
                target_labels.append(LABELS[label_id])
                tokens.append(token)
    targets.append(target_labels)
    texts.append(tokens)

    return targets, texts


def _visualise(module_url, trg_path, prediction_ids, visualised_label,
               visualisation_folder):
    """Generates a .html file comparing the hypothesis/target labels."""
    predictions = [[LABELS[id] for id in ids] for ids in prediction_ids]

    targets, texts = _extract_target_labels(module_url, trg_path)

    assert len(targets) == len(predictions) == len(texts)

    test_data_name = trg_path.split("/")[-1][:-len(".tfrecord")]
    directory = os.path.join(visualisation_folder, test_data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = os.path.join(directory, "%s.html" % visualised_label.lower())
    with open(file_name, "w") as file:
        file.write("%s labels in %s <br>\n" % (visualised_label, trg_path))
        file.write("<font color='green'>Correct labels</font> <br>\n")
        file.write("<font color='blue'>Superfluous labels</font> <br>\n")
        file.write("<font color='red'>Missed labels</font> <br>\n")
        file.write("<br>\n")

        for target_labels, predicted_labels, tokens in zip(
                targets, predictions, texts):
            assert len(target_labels) == len(predicted_labels) == len(tokens)

            for target_label, predicted_label, token in zip(
                    target_labels, predicted_labels, tokens):
                if target_label.endswith(
                        visualised_label) and predicted_label.endswith(
                            visualised_label):
                    file.write("<font color='green'>" + token + "</font>")
                elif target_label.endswith(visualised_label):
                    file.write("<font color='red'>" + token + "</font>")
                elif predicted_label.endswith(visualised_label):
                    file.write("<font color='blue'>" + token + "</font>")
                else:
                    file.write(token)
                file.write(" ")
            file.write("<br>\n")


def _score(module_url, trg_path, prediction_ids, use_strict_mode):
    """Computes the precision, recall and f1 scores of the hypotheses."""
    prediction_labels = [[LABELS[id] for id in ids] for ids in prediction_ids]

    targets, _ = _extract_target_labels(module_url, trg_path)

    if use_strict_mode:
        return classification_report(targets,
                                     prediction_labels,
                                     mode="strict",
                                     scheme=IOB2)
    else:
        return classification_report(targets, prediction_labels)


def main(_):
    for test_data_path in FLAGS.test_data_paths:
        if not test_data_path.endswith(".tfrecord"):
            raise ValueError("The test data must be in .tfrecord format.")

        prediction_ids = _infer(FLAGS.module_url, FLAGS.model_path,
                                test_data_path)
        if FLAGS.visualisation_folder:
            for label in MAIN_LABELS:
                _visualise(FLAGS.module_url, test_data_path, prediction_ids,
                           label, FLAGS.visualisation_folder)

        report = _score(FLAGS.module_url, test_data_path, prediction_ids,
                        FLAGS.strict_eval)
        print("Scores for %s:" % test_data_path)
        print(report)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("model_path")

    app.run(main)

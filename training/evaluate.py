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

from absl import app, flags
import tensorflow as tf
from seqeval.metrics import classification_report

from official.nlp.tasks.tagging import TaggingConfig, TaggingTask, predict
from official.nlp.data import tagging_dataloader
from training.utils import LABELS

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_string("model_path", None, "The path to the trained model.")
flags.DEFINE_multi_string("test_data_paths", [],
                          "The path to the test data in .tfrecord format.")

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


def _score(trg_path, prediction_ids):
    """Computes the precision and recall of the predicted label sequence."""
    prediction_labels = [[LABELS[id] for id in ids] for ids in prediction_ids]

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
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    targets = []
    raw_dataset = tf.data.TFRecordDataset(trg_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    for features in parsed_dataset:
        target = []
        for label in features["label_ids"].numpy():
            if label != -1:
                target.append(LABELS[label])

        if features["sub_sentence_id"][0] == 0:
            targets.append(target)
        else:
            targets[-1].extend(target)

    return classification_report(targets, prediction_labels)


def main(_):
    for test_data_path in FLAGS.test_data_paths:
        prediction_ids = _infer(FLAGS.module_url, FLAGS.model_path,
                                test_data_path)
        report = _score(test_data_path, prediction_ids)
        print(test_data_path)
        print(report)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("model_path")

    app.run(main)

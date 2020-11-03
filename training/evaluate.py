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
import numpy as np
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
import tensorflow as tf
import orbit

from official.nlp.tasks import utils
from official.nlp.tasks.tagging import TaggingConfig, TaggingTask
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


def _predict(task, params, model):
    """Predicts on the input data.

    Similiar to official.nlp.tasks.tagging.predict, but returns the logits
    instead of the final label.

    Args:
        task: A `TaggingTask` object.
        params: A `cfg.DataConfig` object.
        model: A keras.Model.

    Returns:
        A list of tuple. Each tuple contains `sentence_id`, `sub_sentence_id`
            and a list of predicted ids.
    """
    def predict_step(inputs):
        """Replicated prediction calculation."""
        x, y = inputs
        sentence_ids = x.pop("sentence_id")
        sub_sentence_ids = x.pop("sub_sentence_id")
        outputs = task.inference_step(x, model)
        logits = outputs["logits"]
        label_mask = tf.greater_equal(y, 0)
        return dict(logits=logits,
                    label_mask=label_mask,
                    sentence_ids=sentence_ids,
                    sub_sentence_ids=sub_sentence_ids)

    def aggregate_fn(state, outputs):
        """Concatenates model's outputs."""
        if state is None:
            state = []

        for (batch_logits, batch_label_mask, batch_sentence_ids,
             batch_sub_sentence_ids) in zip(outputs["logits"],
                                            outputs["label_mask"],
                                            outputs["sentence_ids"],
                                            outputs["sub_sentence_ids"]):
            batch_probs = tf.keras.activations.softmax(batch_logits)
            for (tmp_prob, tmp_label_mask, tmp_sentence_id,
                 tmp_sub_sentence_id) in zip(batch_probs.numpy(),
                                             batch_label_mask.numpy(),
                                             batch_sentence_ids.numpy(),
                                             batch_sub_sentence_ids.numpy()):

                real_probs = []
                assert len(tmp_prob) == len(tmp_label_mask)
                for i in range(len(tmp_prob)):
                    # Skip the padding label.
                    if tmp_label_mask[i]:
                        real_probs.append(tmp_prob[i])
                state.append(
                    (tmp_sentence_id, tmp_sub_sentence_id, real_probs))

        return state

    dataset = orbit.utils.make_distributed_dataset(
        tf.distribute.get_strategy(), task.build_inputs, params)
    outputs = utils.predict(predict_step, aggregate_fn, dataset)
    return sorted(outputs, key=lambda x: (x[0], x[1]))


def _viterbi(probabilities):
    """"Applies the viterbi algorithm to find the most likely valid label
    sequence.

    Depends on the specific order of labels.
    """
    path_probabilities = [0.0, 0.0, 1.0, 0.0, 0.0]  # Label 3 is "outside"
    path_pointers = []
    for prob_token in probabilities:
        prev_path_probabilities = path_probabilities.copy()
        path_probabilities = [0.0] * 5
        new_pointers = [99] * 5  # An invalid value ensures it will be updated.
        for current_label in range(5):
            for prev_label in range(5):
                prev_prob = prev_path_probabilities[prev_label]
                if (current_label == 1 and prev_label not in [0, 1]) or (
                        current_label == 4 and prev_label not in [3, 4]):
                    prev_prob = 0
                total_prob = prev_prob * prob_token[current_label]
                if total_prob > path_probabilities[current_label]:
                    path_probabilities[current_label] = total_prob
                    new_pointers[current_label] = prev_label
        path_pointers.append(new_pointers)

    most_likely_path = []
    most_likely_end = np.core.fromnumeric.argmax(np.array(path_probabilities))
    while path_pointers:
        pointers = path_pointers.pop()
        most_likely_path.insert(0, most_likely_end)
        most_likely_end = pointers[most_likely_end]
    return most_likely_path


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
    predictions = _predict(task, test_data_config, model)

    merged_probabilities = []
    for _, part_id, predicted_probabilies in predictions:
        if part_id == 0:
            merged_probabilities.append(predicted_probabilies)
        else:
            merged_probabilities[-1].extend(predicted_probabilies)

    merged_predictions = []
    for probabilities in merged_probabilities:
        prediction = _viterbi(probabilities)
        merged_predictions.append(prediction)

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

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
from training.utils import (ADDITIONAL_LABELS, LABELS, LABEL_OUTSIDE,
                            MAIN_LABELS, create_tokenizer_from_hub_module,
                            split_into_words,
                            get_labeled_text_from_linkfragment, get_documents,
                            get_labeled_text_from_document)

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_string("model_path", None, "The path to the trained model.")
flags.DEFINE_multi_string("tfrecord_paths", [],
                          "The paths to the test data in .tfrecord format.")
flags.DEFINE_multi_string(
    "raw_paths", [],
    "The paths to the test data in its original .binproto or .lftxt format.")
flags.DEFINE_string(
    "visualisation_folder", None,
    "If set, a comparison of the target/hypothesis labeling is saved in .html"
    " format")
flags.DEFINE_boolean(
    "strict_eval", False,
    "Only used for scoring. If True, a label must not begin with an 'I-' tag.")
flags.DEFINE_boolean(
    "train_with_additional_labels", False,
    "Needs to be set if the flags other than address/phone were used for"
    " training, too.")

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
    def _predict_step(inputs):
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

    def _aggregate_fn(state, outputs):
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
    outputs = utils.predict(_predict_step, _aggregate_fn, dataset)
    return sorted(outputs, key=lambda x: (x[0], x[1]))


def _viterbi(probabilities, train_with_additional_labels):
    """"Applies the viterbi algorithm.

    This searches for the most likely valid label sequence.
    Depends on the specific order of labels.
    """
    labels = LABELS
    if train_with_additional_labels:
        labels += ADDITIONAL_LABELS
    path_probabilities = np.zeros(len(labels))
    label_outside_index = labels.index(LABEL_OUTSIDE)
    path_probabilities[label_outside_index] = 1.0
    label_id_map = {label: i for i, label in enumerate(labels)}
    path_pointers = []
    for prob_token in probabilities:
        prev_path_probabilities = path_probabilities.copy()
        path_probabilities = np.zeros(len(labels))
        new_pointers = [len(labels) + 1] * len(
            labels)  # An invalid value ensures it will be updated.
        for current_label_id in range(len(labels)):
            current_label_name = labels[current_label_id]
            if current_label_name.startswith("I-"):
                current_main_label_name = current_label_name[2:]
                valid_prev_label_names = [("B-%s" % current_main_label_name),
                                          ("I-%s" % current_main_label_name)]
                mask = np.zeros(len(labels))
                for prev_label_name in valid_prev_label_names:
                    prev_label_id = label_id_map[prev_label_name]
                    mask[prev_label_id] = 1
                masked_prev_path_probabilities = prev_path_probabilities * mask
            else:
                masked_prev_path_probabilities = prev_path_probabilities
            total_prob = masked_prev_path_probabilities * prob_token[
                current_label_id]
            max_prob_index = np.argmax(total_prob)
            path_probabilities[current_label_id] = total_prob[max_prob_index]
            new_pointers[current_label_id] = max_prob_index
        path_pointers.append(new_pointers)

    most_likely_path = []
    most_likely_end = np.core.fromnumeric.argmax(np.array(path_probabilities))
    while path_pointers:
        pointers = path_pointers.pop()
        most_likely_path.insert(0, most_likely_end)
        most_likely_end = pointers[most_likely_end]
    return most_likely_path


def _infer(module_url, model_path, test_data_path,
           train_with_additional_labels):
    """Computes the predicted label sequence using the trained model."""
    test_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=test_data_path,
        seq_length=128,
        global_batch_size=64,
        is_training=False,
        include_sentence_id=True,
        drop_remainder=False)
    labels = LABELS
    if train_with_additional_labels:
        labels += ADDITIONAL_LABELS
    config = TaggingConfig(hub_module_url=module_url, class_names=labels)
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
        prediction = _viterbi(probabilities, train_with_additional_labels)
        merged_predictions.append(prediction)

    return merged_predictions


def _remove_whitespace_and_parse(text, tokenizer):
    """Removes all whitespace and some special characters.

    The tokenizer discards some utf-8 characters, such as the right-to-left
    indicator. Applying the tokenizer is slow, but the safest way to guarantee
    consistent behaviour.
    """
    return "".join(split_into_words(text, tokenizer))


def _update_characterwise_target_labels(tokenizer, prefix, labeled_text,
                                        suffix, label,
                                        characterwise_target_labels,
                                        characters):
    """Updates target_labels and characters w.r.t. the given text and label."""
    prefix_without_whitespace = _remove_whitespace_and_parse(prefix, tokenizer)
    characters.extend(prefix_without_whitespace)
    characterwise_target_labels.extend([LABEL_OUTSIDE] *
                                       len(prefix_without_whitespace))

    labeled_text_without_whitespace = _remove_whitespace_and_parse(
        labeled_text, tokenizer)
    characters.extend(labeled_text_without_whitespace)
    if len(labeled_text_without_whitespace) > 0:
        characterwise_target_labels.append("B-%s" % label)
        characterwise_target_labels.extend(
            ["I-%s" % label] * (len(labeled_text_without_whitespace) - 1))

    suffix_without_whitespace = _remove_whitespace_and_parse(suffix, tokenizer)
    characters.extend(suffix_without_whitespace)
    characterwise_target_labels.extend([LABEL_OUTSIDE] *
                                       len(suffix_without_whitespace))


def _extract_characterwise_target_labels_from_proto(path, tokenizer):
    """Extracts a label for each character from the given .binproto file."""
    characterwise_target_labels_per_sentence = []
    characters_per_sentence = []
    for document in get_documents(path):
        characterwise_target_labels = []
        characters = []
        for prefix, labeled_text, label in get_labeled_text_from_document(
                document, only_main_labels=True):
            _update_characterwise_target_labels(tokenizer, prefix,
                                                labeled_text, "", label,
                                                characterwise_target_labels,
                                                characters)

        characterwise_target_labels_per_sentence.append(
            characterwise_target_labels)
        characters_per_sentence.append(characters)
    return characterwise_target_labels_per_sentence, characters_per_sentence


def _extract_characterwise_target_labels_from_lftxt(path, tokenizer):
    """Extracts a label for each character from the given .lftxt file."""
    characterwise_target_labels_per_sentence = []
    characters_per_sentence = []
    with open(path, "r") as src_file:
        for (prefix, labeled_text,
             suffix), _, label in get_labeled_text_from_linkfragment(src_file):
            characterwise_target_labels = []
            characters = []

            _update_characterwise_target_labels(tokenizer, prefix,
                                                labeled_text, suffix, label,
                                                characterwise_target_labels,
                                                characters)

            characterwise_target_labels_per_sentence.append(
                characterwise_target_labels)
            characters_per_sentence.append(characters)

    return characterwise_target_labels_per_sentence, characters_per_sentence


def _visualise(test_name, characterwise_target_labels_per_sentence,
               characterwise_predicted_labels_per_sentence,
               characters_per_sentence, words_per_sentence, visualised_label,
               visualisation_folder):
    """Generates a .html file comparing the hypothesis/target labels."""
    assert len(characterwise_target_labels_per_sentence) == len(
        characterwise_predicted_labels_per_sentence) == len(
            characters_per_sentence)

    directory = os.path.join(visualisation_folder, test_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = os.path.join(directory, "%s.html" % visualised_label.lower())
    with open(file_name, "w") as file:
        file.write("%s labels in %s <br>\n" % (visualised_label, test_name))
        file.write("<font color='green'>Correct labels</font> <br>\n")
        file.write("<font color='blue'>Superfluous labels</font> <br>\n")
        file.write("<font color='red'>Missed labels</font> <br>\n")
        file.write("<br>\n")

        for i, (characterwise_target_labels, characterwise_predicted_labels,
                characters, words) in enumerate(
                    zip(characterwise_target_labels_per_sentence,
                        characterwise_predicted_labels_per_sentence,
                        characters_per_sentence, words_per_sentence)):
            characterwise_target_labels_length = len(
                characterwise_target_labels)
            characterwise_predicted_labels_length = len(
                characterwise_predicted_labels)
            characters_length = len(characters)
            assert (
                characterwise_target_labels_length ==
                characterwise_predicted_labels_length == characters_length
            ), ("Hypotheses/targets have different lengths: %d, %d, %d"
                " (sentence %d)") % (characterwise_target_labels_length,
                                     characterwise_predicted_labels_length,
                                     characters_length, i)

            word_index = 0
            word_position = 0
            for target_label, predicted_label, character in zip(
                    characterwise_target_labels,
                    characterwise_predicted_labels, characters):
                if target_label.endswith(
                        visualised_label) and predicted_label.endswith(
                            visualised_label):
                    file.write("<font color='green'>" + character + "</font>")
                elif target_label.endswith(visualised_label):
                    file.write("<font color='red'>" + character + "</font>")
                elif predicted_label.endswith(visualised_label):
                    file.write("<font color='blue'>" + character + "</font>")
                else:
                    file.write(character)

                word_position += 1
                if word_position == len(words[word_index]):
                    word_index += 1
                    word_position = 0
                    file.write(" ")

            file.write("<br>\n")


def _score(characterwise_target_labels_per_sentence,
           characterwise_predicted_labels_per_sentence, use_strict_mode):
    """Computes the precision, recall and f1 scores of the hypotheses."""
    if use_strict_mode:
        return classification_report(
            characterwise_target_labels_per_sentence,
            characterwise_predicted_labels_per_sentence,
            mode="strict",
            scheme=IOB2)
    else:
        return classification_report(
            characterwise_target_labels_per_sentence,
            characterwise_predicted_labels_per_sentence)


def _extract_words_from_proto(path, tokenizer):
    """Extracts all words from the .binproto file."""
    words_per_sentence = []
    for document in get_documents(path):
        words = split_into_words(document.text, tokenizer)
        words_per_sentence.append(words)
    return words_per_sentence


def _extract_words_from_lftxt(path, tokenizer):
    """Extracts all words from the given .lftxt file."""
    words_per_sentence = []
    with open(path, "r") as src_file:
        for _, text_without_braces, _ in get_labeled_text_from_linkfragment(
                src_file):
            words = split_into_words(text_without_braces, tokenizer)
            words_per_sentence.append(words)
    return words_per_sentence


def _transform_wordwise_labels_to_characterwise_labels(
        words_per_sentence, predicted_label_ids_per_sentence):
    """Duplicates the labels such that each character is assigned a label.

    For "B-" labels, only the first character of the word is assigned the "B-"
    label, all other characters are assigned the corresponding "I-" label.
    """
    characterwise_predicted_label_ids_per_sentence = []
    for words, predicted_label_ids in zip(words_per_sentence,
                                          predicted_label_ids_per_sentence):
        characterwise_predicted_label_ids = []

        assert len(words) == len(predicted_label_ids)
        for word, label_id in zip(words, predicted_label_ids):
            if LABELS[label_id].startswith("B-"):
                characterwise_predicted_label_ids.append(label_id)
                characterwise_predicted_label_ids.extend([label_id + 1] *
                                                         (len(word) - 1))
            else:
                characterwise_predicted_label_ids.extend([label_id] *
                                                         len(word))
        characterwise_predicted_label_ids_per_sentence.append(
            characterwise_predicted_label_ids)
    return characterwise_predicted_label_ids_per_sentence


def _convert_label_ids_to_names(label_ids_per_sentence):
    labels = LABELS + ADDITIONAL_LABELS
    label_names_per_sentence = [[labels[label_id] for label_id in label_ids]
                                for label_ids in label_ids_per_sentence]
    return label_names_per_sentence


def main(_):
    if len(FLAGS.tfrecord_paths) != len(FLAGS.raw_paths):
        raise ValueError("The number of tfrecord and raw paths must be equal.")

    for tfrecord_path, raw_path in zip(FLAGS.tfrecord_paths, FLAGS.raw_paths):
        if not tfrecord_path.endswith(".tfrecord"):
            raise ValueError("The test data must be in .tfrecord format.")
        test_name = raw_path.split("/")[-1].split(".")[0]
        tokenizer = create_tokenizer_from_hub_module(FLAGS.module_url)

        predicted_label_ids_per_sentence = _infer(
            FLAGS.module_url, FLAGS.model_path, tfrecord_path,
            FLAGS.train_with_additional_labels)

        if "proto" in raw_path:
            words_per_sentence = _extract_words_from_proto(raw_path, tokenizer)
        else:
            words_per_sentence = _extract_words_from_lftxt(raw_path, tokenizer)

        characterwise_predicted_label_ids_per_sentence = (
            _transform_wordwise_labels_to_characterwise_labels(
                words_per_sentence, predicted_label_ids_per_sentence))

        characterwise_predicted_label_names_per_sentence = (
            _convert_label_ids_to_names(
                characterwise_predicted_label_ids_per_sentence))

        if "proto" in raw_path:
            (characterwise_target_labels_per_sentence,
             characters_per_sentence) = (
                 _extract_characterwise_target_labels_from_proto(
                     raw_path, tokenizer))
        else:
            (characterwise_target_labels_per_sentence,
             characters_per_sentence) = (
                 _extract_characterwise_target_labels_from_lftxt(
                     raw_path, tokenizer))

        if FLAGS.visualisation_folder:
            for visualised_label in MAIN_LABELS:
                _visualise(test_name, characterwise_target_labels_per_sentence,
                           characterwise_predicted_label_names_per_sentence,
                           characters_per_sentence, words_per_sentence,
                           visualised_label, FLAGS.visualisation_folder)

        report = _score(characterwise_target_labels_per_sentence,
                        characterwise_predicted_label_names_per_sentence,
                        FLAGS.strict_eval)
        print("Scores for %s:" % test_name)
        print(report)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("model_path")

    app.run(main)

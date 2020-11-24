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
from enum import Enum
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
                            MAIN_LABELS, LabeledExample,
                            create_tokenizer_from_hub_module, split_into_words,
                            get_labeled_text_from_linkfragment, get_documents,
                            get_labeled_text_from_document)

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_string("model_path", None, "The path to the trained model.")
flags.DEFINE_multi_string(
    "input_paths", [],
    "The paths to the test data in .tfrecord format (to be labeled) or a folder"
    " containing .lftxt files with precomputed labels.")
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
flags.DEFINE_string("output_directory", None,
                    "If given, the hypotheses are saved to this directory.")

FLAGS = flags.FLAGS

LabelType = Enum("LabelType", "OUTSIDE BEGINNING INSIDE")


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
            if _is_label_type(current_label_name, LabelType.INSIDE):
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


def _update_characterwise_target_labels(tokenizer, labeled_example,
                                        characterwise_target_labels):
    """Updates target_labels and characters w.r.t. the given text and label."""
    prefix_without_whitespace = _remove_whitespace_and_parse(
        labeled_example.prefix, tokenizer)
    labeled_text_without_whitespace = _remove_whitespace_and_parse(
        labeled_example.selection, tokenizer)
    if len(labeled_text_without_whitespace) > 0:
        start = len(prefix_without_whitespace)
        end = start + len(labeled_text_without_whitespace) - 1
        characterwise_target_labels[start] = "B-%s" % labeled_example.label
        characterwise_target_labels[start + 1:end +
                                    1] = ["I-%s" % labeled_example.label
                                          ] * (end - start)


def _extract_characterwise_target_labels_from_proto(path, tokenizer):
    """Extracts a label for each character from the given .binproto file."""
    characterwise_target_labels_per_sentence = []
    characters_per_sentence = []
    for document in get_documents(path):
        characters = _remove_whitespace_and_parse(document.text, tokenizer)
        characterwise_target_labels = [LABEL_OUTSIDE] * len(characters)
        total_prefix = ""
        for labeled_example in get_labeled_text_from_document(
                document, only_main_labels=True):
            assert labeled_example.suffix == ""
            total_prefix += labeled_example.prefix
            labeled_example.prefix = total_prefix
            _update_characterwise_target_labels(tokenizer, labeled_example,
                                                characterwise_target_labels)
            total_prefix += labeled_example.selection

        characterwise_target_labels_per_sentence.append(
            characterwise_target_labels)
        characters_per_sentence.append(characters)
    return characterwise_target_labels_per_sentence, characters_per_sentence


def _extract_characterwise_target_labels_from_lftxt(path, tokenizer,
                                                    merge_identical_sentences):
    """Extracts a label for each character from the given .lftxt file."""
    characterwise_target_labels_per_sentence = []
    characters_per_sentence = []
    characterwise_target_labels = []
    characters = []
    prev_text = ""
    for labeled_example in get_labeled_text_from_linkfragment(path):
        if merge_identical_sentences and (prev_text
                                          == labeled_example.complete_text):
            # The last entry is updated, it will be added again.
            del characterwise_target_labels_per_sentence[-1]
            del characters_per_sentence[-1]
        else:
            characters = _remove_whitespace_and_parse(
                labeled_example.complete_text, tokenizer)
            characterwise_target_labels = [LABEL_OUTSIDE] * len(characters)

        _update_characterwise_target_labels(tokenizer, labeled_example,
                                            characterwise_target_labels)

        characterwise_target_labels_per_sentence.append(
            characterwise_target_labels)
        characters_per_sentence.append(characters)
        prev_text = labeled_example.complete_text

    return characterwise_target_labels_per_sentence, characters_per_sentence


def _visualise(test_name, characterwise_target_labels_per_sentence,
               characterwise_predicted_labels_per_sentence,
               characters_per_sentence, words_per_sentence, visualised_label,
               visualisation_folder):
    """Generates a .html file comparing the hypothesis/target labels."""
    assert len(characterwise_target_labels_per_sentence) == len(
        characterwise_predicted_labels_per_sentence) == len(
            characters_per_sentence)
    number_of_sentences = len(characterwise_target_labels_per_sentence)

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

        for i in range(number_of_sentences):
            characterwise_target_labels = (
                characterwise_target_labels_per_sentence[i])
            characterwise_predicted_labels = (
                characterwise_predicted_labels_per_sentence[i])
            characters = characters_per_sentence[i]
            words = words_per_sentence[i]

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


def _extract_words_from_lftxt(path, tokenizer, merge_identical_sentences):
    """Extracts all words as defined by the tokenizer from the given .lftxt
    file."""
    words_per_sentence = []
    prev_text = ""
    for labeled_example in get_labeled_text_from_linkfragment(path):
        if merge_identical_sentences and (prev_text
                                          == labeled_example.complete_text):
            continue
        words = split_into_words(labeled_example.complete_text, tokenizer)
        prev_text = labeled_example.complete_text
        words_per_sentence.append(words)
    return words_per_sentence


def _is_label_type(label_name, label_type):
    """Checks whether the label is of the specified type."""
    if label_name == LABEL_OUTSIDE:
        real_label_type = LabelType.OUTSIDE
    elif label_name.startswith("B-"):
        real_label_type = LabelType.BEGINNING
    else:
        assert label_name.startswith("I-")
        real_label_type = LabelType.INSIDE
    return label_type == real_label_type


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
            if _is_label_type(LABELS[label_id], LabelType.BEGINNING):
                characterwise_predicted_label_ids += [
                    label_id
                ] + [label_id + 1] * (len(word) - 1)
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


def _unescape_backslashes(labeled_example):
    return LabeledExample(
        prefix=labeled_example.prefix.replace("\\\\", "\\"),
        selection=labeled_example.selection.replace("\\\\", "\\"),
        suffix=labeled_example.suffix.replace("\\\\", "\\"),
        complete_text=labeled_example.complete_text.replace("\\\\", "\\"),
        label=labeled_example.label)


def _get_predictions_from_lf_directory(lf_directory, raw_path, tokenizer):
    """Gets the characterwise labels from all .lftxt files in the directory.

    Args:
        lf_directory: Path to the directory. All contained .lftxt files are
            parsed.
        raw_path: Path to the file containing all sentences as they are used as
            the input for inference. Necessary to get the correct sentence
            order for the evaluation.
        tokenizer: Tokenizer. Necessary to split the text into words and to
            remove whitespace characters.

    Returns:
        List of characterwise target labels per sentence.
    """
    labeled_sentences = dict()  # Map sentences to their labels.

    if raw_path.endswith(".lftxt"):
        prev_text = ""
        for labeled_example in get_labeled_text_from_linkfragment(raw_path):
            if prev_text == labeled_example.complete_text:
                continue
            prev_text = labeled_example.complete_text
            characters = _remove_whitespace_and_parse(
                labeled_example.complete_text, tokenizer)

            assert labeled_example.complete_text not in labeled_sentences
            labeled_sentences[characters] = [LABEL_OUTSIDE] * len(characters)
    else:
        for document in get_documents(raw_path):
            characters = _remove_whitespace_and_parse(document.text, tokenizer)
            labeled_sentences[characters] = [LABEL_OUTSIDE] * len(characters)

    for file_name in os.listdir(lf_directory):
        if not file_name.endswith(".lftxt"):
            continue

        for labeled_example in get_labeled_text_from_linkfragment(
                os.path.join(lf_directory, file_name)):
            if labeled_example.label == LABEL_OUTSIDE:
                continue
            labeled_example = _unescape_backslashes(labeled_example)
            prefix_length = len(
                _remove_whitespace_and_parse(labeled_example.prefix,
                                             tokenizer))
            label_length = len(
                _remove_whitespace_and_parse(labeled_example.selection,
                                             tokenizer))
            assert label_length > 0

            characters = _remove_whitespace_and_parse(
                labeled_example.complete_text, tokenizer)
            # If the .lftxt file was generated as the output of another models
            # prediction, the tokenizer will have lowercased the [UNK] token.
            characters = characters.replace("[unk]", "[UNK]")
            characterwise_labels = labeled_sentences[characters]
            assert characterwise_labels[prefix_length] in [
                LABEL_OUTSIDE,
                "B-%s" % labeled_example.label.upper()
            ]
            characterwise_labels[
                prefix_length] = "B-%s" % labeled_example.label.upper()
            assert all([
                label
                in [LABEL_OUTSIDE,
                    "I-%s" % labeled_example.label.upper()]
                for label in characterwise_labels[prefix_length +
                                                  1:prefix_length +
                                                  label_length]
            ])
            characterwise_labels[prefix_length + 1:prefix_length +
                                 label_length] = [
                                     "I-%s" % labeled_example.label.upper()
                                 ] * (label_length - 1)

    # The order is important, because it controls which label sequences are
    # compared in the evaluation. Python 3.6+ preserves the insertion order.
    return list(labeled_sentences.values())


def _extract_words(raw_path, tokenizer):
    """Extracts all words as defined by the tokenizer for all sentences."""
    if "proto" in raw_path:
        return _extract_words_from_proto(raw_path, tokenizer)
    else:
        return _extract_words_from_lftxt(raw_path,
                                         tokenizer,
                                         merge_identical_sentences=True)


def _get_characterwise_predicted_label_names(module_url, model_path,
                                             input_path,
                                             train_with_additional_labels,
                                             words_per_sentence, raw_path,
                                             tokenizer):
    """Extracts the characterwise label names."""
    if input_path.endswith(".tfrecord"):
        predicted_label_ids_per_sentence = _infer(
            module_url, model_path, input_path, train_with_additional_labels)

        characterwise_predicted_label_ids_per_sentence = (
            _transform_wordwise_labels_to_characterwise_labels(
                words_per_sentence, predicted_label_ids_per_sentence))

        characterwise_predicted_label_names_per_sentence = (
            _convert_label_ids_to_names(
                characterwise_predicted_label_ids_per_sentence))
    else:
        characterwise_predicted_label_names_per_sentence = (
            _get_predictions_from_lf_directory(input_path, raw_path,
                                               tokenizer))

    return characterwise_predicted_label_names_per_sentence


def _extract_characterwise_target_labels(raw_path, tokenizer):
    """Extracts a label for each character."""
    if "proto" in raw_path:
        return _extract_characterwise_target_labels_from_proto(
            raw_path, tokenizer)
    else:
        return _extract_characterwise_target_labels_from_lftxt(
            raw_path, tokenizer, merge_identical_sentences=True)


def _get_text_from_character_indices(words, start, end):
    """Returns text between the start/end indices which do not count spaces."""
    text = ""
    accumulated_word_length = 0
    for word in words:
        accumulated_word_length += len(word)
        if accumulated_word_length <= start:
            continue
        elif end is not None and accumulated_word_length > end + 1:
            return text
        if text != "":
            text += " "
        text += word
    return text


def _save_as_linkfragment(words, label_start, label_end, label, file):
    """Writes a linkfragment to the file."""
    prefix = _get_text_from_character_indices(words, 0, label_start - 1)
    labelled_text = _get_text_from_character_indices(words, label_start,
                                                     label_end)
    suffix = _get_text_from_character_indices(words, label_end + 1, None)

    file.write("%s {{{%s}}} %s\t%s\n" %
               (prefix, labelled_text, suffix, label.lower()))


def _save_predictions(output_directory, test_name,
                      characterwise_predicted_label_names_per_sentence,
                      words_per_sentence):
    """Saves the hypotheses to an .lftxt file."""
    with open(os.path.join(output_directory, "%s.lftxt" % test_name),
              "w") as output_file:
        for (characterwise_predicted_label_names,
             words) in zip(characterwise_predicted_label_names_per_sentence,
                           words_per_sentence):
            label_start = 0
            label = None
            saved_at_least_once = False
            for i, label_name in enumerate(
                    characterwise_predicted_label_names):
                if label_name == LABEL_OUTSIDE:
                    if label is not None:
                        _save_as_linkfragment(words, label_start, i - 1, label,
                                              output_file)
                        label = None
                        saved_at_least_once = True
                elif label_name.startswith("B-"):
                    if label is not None:
                        _save_as_linkfragment(words, label_start, i - 1, label,
                                              output_file)
                        saved_at_least_once = True
                    label = label_name[len("B-"):]
                    label_start = i
                else:
                    assert label_name == "I-%s" % label
            if not saved_at_least_once:
                _save_as_linkfragment(words, 0, -1, "OUTSIDE", output_file)


def main(_):
    if len(FLAGS.input_paths) != len(FLAGS.raw_paths):
        raise ValueError("The number of inputs and raw paths must be equal.")

    for input_path, raw_path in zip(FLAGS.input_paths, FLAGS.raw_paths):
        test_name = os.path.splitext(os.path.basename(raw_path))[0]
        tokenizer = create_tokenizer_from_hub_module(FLAGS.module_url)

        words_per_sentence = _extract_words(raw_path, tokenizer)

        characterwise_predicted_label_names_per_sentence = (
            _get_characterwise_predicted_label_names(
                FLAGS.module_url, FLAGS.model_path, input_path,
                FLAGS.train_with_additional_labels, words_per_sentence,
                raw_path, tokenizer))

        (characterwise_target_labels_per_sentence,
         characters_per_sentence) = _extract_characterwise_target_labels(
             raw_path, tokenizer)

        if FLAGS.output_directory:
            _save_predictions(
                FLAGS.output_directory, test_name,
                characterwise_predicted_label_names_per_sentence,
                words_per_sentence)

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

    app.run(main)

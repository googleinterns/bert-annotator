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
from collections import OrderedDict
from absl import app, flags
import numpy as np
from official.nlp.data import tagging_data_lib
from official.nlp.tasks import utils
from official.nlp.tasks.tagging import TaggingConfig, TaggingTask
from official.nlp.data import tagging_dataloader
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
import tensorflow as tf
from tensorflow.io.gfile import GFile
import orbit
from com_google_research_bert import tokenization

from training.utils import (ADDITIONAL_LABELS, LABELS, LABEL_ID_MAP,
                            LABEL_OUTSIDE, MAIN_LABELS, LabeledExample,
                            add_tfrecord_label,
                            create_tokenizer_from_hub_module,
                            remove_whitespace_and_parse, write_example_to_file)
from training.file_reader import get_file_reader
import protocol_buffer.documents_pb2 as proto_documents

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
flags.DEFINE_multi_enum(
    "save_output_formats", [], ["lftxt", "binproto", "tfrecord"],
    "If set, the hypotheses are saved in the corresponding formats.")
flags.DEFINE_string("output_directory", None,
                    "Controls where to save the hypotheses.")
flags.DEFINE_integer(
    "moving_window_overlap", 20, "The size of the overlap for a moving window."
    " Setting it to zero restores the default behaviour of hard splitting.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximal sequence length. Longer sequences are split.")
flags.DEFINE_integer("batch_size", 64, "The number of samples per batch.")
flags.DEFINE_string(
    "tpu_address", None,
    "The internal address of the TPU node, including 'grpc://'. If not set, no"
    " tpu is used.")
flags.DEFINE_multi_integer(
    "unlabeled_sentence_filters", [1],
    "For each entry, a copy of the hypothesis is saved (if activated). Only"
    " every nth sentence without any labels is kept, the other unlabeled"
    " sentences are dropped. Sentences with at least one predicted label are"
    " always kept.")

FLAGS = flags.FLAGS

LabelType = Enum("LabelType", "OUTSIDE BEGINNING INSIDE")


def _assert_same_length(sequences, sentence_id=None):
    lengths = [len(sequence) for sequence in sequences]
    if len(lengths) == 0:
        return
    if sentence_id is not None:
        additional_information = "(sentence id %d)" % sentence_id
    else:
        additional_information = ""

    for length in lengths[1:]:
        assert length == lengths[
            0], "Not all sequences have the same length: %s %s" % (
                str(lengths), additional_information)


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
                _assert_same_length([tmp_prob, tmp_label_mask],
                                    tmp_sentence_id)
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
    path_probabilities = np.full(len(labels), -np.inf)
    label_outside_index = labels.index(LABEL_OUTSIDE)
    path_probabilities[label_outside_index] = 0.0
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
                mask = np.full(len(labels), True)
                for prev_label_name in valid_prev_label_names:
                    prev_label_id = LABEL_ID_MAP[prev_label_name]
                    mask[prev_label_id] = False
                masked_prev_path_probabilities = prev_path_probabilities.copy()
                masked_prev_path_probabilities[mask] = -np.inf
            else:
                masked_prev_path_probabilities = prev_path_probabilities
            total_prob = masked_prev_path_probabilities + np.log(
                prob_token[current_label_id])
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


def _get_model_and_task(module_url, model_path, train_with_additional_labels):
    """Returns the loaded model and corresponding task."""
    labels = LABELS
    if train_with_additional_labels:
        labels += ADDITIONAL_LABELS
    config = TaggingConfig(hub_module_url=module_url, class_names=labels)
    task = TaggingTask(config)
    if model_path:
        model = task.build_model()
        model.load_weights(model_path)
    else:
        model = None
    return model, task


def _infer(model, task, test_data_path, train_with_additional_labels,
           batch_size):
    """Computes the predicted label sequence using the trained model."""
    test_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=test_data_path,
        seq_length=128,
        global_batch_size=batch_size,
        is_training=False,
        include_sentence_id=True,
        drop_remainder=False)
    predictions = _predict(task, test_data_config, model)

    merged_probabilities = []
    for _, part_id, predicted_probabilies in predictions:
        if part_id == 0:
            merged_probabilities.append(predicted_probabilies)
        else:
            merged_probabilities[-1].extend(predicted_probabilies)

    merged_predictions = []
    for i, probabilities in enumerate(merged_probabilities):
        assert not np.isnan(probabilities).any(), (
            "There was an error during decoding. Try reducing the batch size."
            " First error in sentence %d" % i)
        prediction = _viterbi(probabilities, train_with_additional_labels)
        merged_predictions.append(prediction)

    return merged_predictions


def _visualise(test_name, characterwise_target_labels_per_sentence,
               characterwise_predicted_labels_per_sentence,
               characters_per_sentence, words_per_sentence, visualised_label,
               visualisation_folder):
    """Generates a .html file comparing the hypothesis/target labels."""
    _assert_same_length([
        characterwise_target_labels_per_sentence,
        characterwise_predicted_labels_per_sentence, characters_per_sentence
    ])
    number_of_sentences = len(characterwise_target_labels_per_sentence)

    directory = os.path.join(visualisation_folder, test_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = os.path.join(directory, "%s.html" % visualised_label.lower())
    with GFile(file_name, "w") as file:
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
    _assert_same_length([words_per_sentence, predicted_label_ids_per_sentence])
    for i, (words, predicted_label_ids) in enumerate(
            zip(words_per_sentence, predicted_label_ids_per_sentence)):
        characterwise_predicted_label_ids = []

        _assert_same_length([words, predicted_label_ids], i)
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
    labeled_sentences = OrderedDict()  # Map sentences to their labels.

    _, characters_per_sentence = get_file_reader(
        raw_path).get_characterwise_target_labels(tokenizer)
    characters_without_whitespace_per_sentence = [
        remove_whitespace_and_parse(characters, tokenizer)
        for characters in characters_per_sentence
    ]
    labeled_sentences = {
        characters_without_whitespace:
        [LABEL_OUTSIDE] * len(characters_without_whitespace)
        for characters_without_whitespace in
        characters_without_whitespace_per_sentence
    }

    for file_name in os.listdir(lf_directory):
        if not file_name.endswith(".lftxt"):
            continue

        for labeled_example in get_file_reader(
                os.path.join(lf_directory,
                             file_name)).get_labeled_text(tokenizer):
            if labeled_example.label == LABEL_OUTSIDE:
                continue
            labeled_example = _unescape_backslashes(labeled_example)
            prefix_length = len(
                remove_whitespace_and_parse(labeled_example.prefix, tokenizer))
            label_length = len(
                remove_whitespace_and_parse(labeled_example.selection,
                                            tokenizer))
            assert label_length > 0

            characters = remove_whitespace_and_parse(
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
    # compared in the evaluation. The usage of OrderedDict allows this.
    return list(labeled_sentences.values())


def _infer_characterwise_label_names(model, task, input_path,
                                     train_with_additional_labels,
                                     words_per_sentence, raw_path, tokenizer,
                                     batch_size):
    """Extracts the characterwise label names."""
    if input_path.endswith(".tfrecord"):
        predicted_label_ids_per_sentence = _infer(
            model, task, input_path, train_with_additional_labels, batch_size)

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


def _get_word_indices_from_character_indices(words, char_start, char_end):
    """Converts the indices without whitespace to word indices."""
    if char_start == 0 and char_end == -1:
        return 0, -1  # End < start as both are inclusive.
    word_start = None
    word_end = None
    accumulated_word_length = 0
    for i, word in enumerate(words):
        if accumulated_word_length == char_start:
            word_start = i
        accumulated_word_length += len(word)
        if accumulated_word_length - 1 == char_end:
            word_end = i
    if accumulated_word_length == char_start:
        word_start = len(words)
    if char_end is None:
        word_end = len(words) - 1
    assert word_start is not None and word_end is not None
    return word_start, word_end


def _get_text_from_character_indices(words, start, end):
    """Returns text between the start/end indices which do not count spaces."""
    word_start, word_end = _get_word_indices_from_character_indices(
        words, start, end)
    return " ".join(words[word_start:word_end + 1])


def _save_as_linkfragment(words, label_start, label_end, label, file):
    """Writes a linkfragment to the file."""
    prefix = _get_text_from_character_indices(words, 0, label_start - 1)
    labelled_text = _get_text_from_character_indices(words, label_start,
                                                     label_end)
    suffix = _get_text_from_character_indices(words, label_end + 1, None)

    file.write("%s {{{%s}}} %s\t%s\n" %
               (prefix, labelled_text, suffix, label.lower()))


def _save_predictions_as_lftxt(
        output_directory, file_name,
        characterwise_predicted_label_names_per_sentence, words_per_sentence):
    """Saves the hypotheses to an .lftxt file."""
    with GFile(os.path.join(output_directory, file_name), "w") as output_file:
        for (characterwise_predicted_label_names,
             words) in zip(characterwise_predicted_label_names_per_sentence,
                           words_per_sentence):
            label_start = 0
            label = None
            saved_at_least_once = False
            for i, label_name in enumerate(
                    characterwise_predicted_label_names):
                if _is_label_type(label_name, LabelType.OUTSIDE):
                    if label is not None:
                        _save_as_linkfragment(words, label_start, i - 1, label,
                                              output_file)
                        label = None
                        saved_at_least_once = True
                elif _is_label_type(label_name, LabelType.BEGINNING):
                    if label is not None:
                        _save_as_linkfragment(words, label_start, i - 1, label,
                                              output_file)
                        saved_at_least_once = True
                    label = label_name[len("B-"):]
                    label_start = i
                else:
                    assert label_name == "I-%s" % label
            # If the label goes until the very end of the sentence.
            if label is not None:
                _save_as_linkfragment(
                    words, label_start,
                    len(characterwise_predicted_label_names) - 1, label,
                    output_file)
                saved_at_least_once = True
            if not saved_at_least_once:
                _save_as_linkfragment(words, 0, -1, "OUTSIDE", output_file)


def _save_predictions_as_binproto(
        output_directory, file_name,
        characterwise_predicted_label_names_per_sentence, words_per_sentence):
    """Saves the hypotheses to a .binproto file."""
    def _add_labeled_span(label_start_char, label_end_char, label, words,
                          proto_labeled_spans):
        """Adds a new labeled span to the list."""
        label_start, label_end = (_get_word_indices_from_character_indices(
            words, label_start_char, label_end_char))
        proto_labeled_spans.labeled_span.add(token_start=label_start,
                                             token_end=label_end,
                                             label=label)

    with GFile(os.path.join(output_directory, file_name), "wb") as output_file:
        documents = proto_documents.Documents()
        for (characterwise_predicted_label_names,
             words) in zip(characterwise_predicted_label_names_per_sentence,
                           words_per_sentence):
            document = documents.documents.add()
            document.text = " ".join(words)
            token_start = 0
            for word in words:
                document.token.add(start=token_start,
                                   end=token_start + len(word) - 1,
                                   word=word)
                token_start += len(word) + 1

            label_start_char = 0
            label = None
            proto_labeled_spans = document.labeled_spans["lucid"]
            for i, label_name in enumerate(
                    characterwise_predicted_label_names):
                if _is_label_type(label_name, LabelType.OUTSIDE):
                    if label is not None:
                        _add_labeled_span(label_start_char, i - 1, label,
                                          words, proto_labeled_spans)
                        label = None
                elif _is_label_type(label_name, LabelType.BEGINNING):
                    if label is not None:
                        _add_labeled_span(label_start_char, i - 1, label,
                                          words, proto_labeled_spans)
                    label = label_name[len("B-"):]
                    label_start_char = i
                else:
                    assert label_name == "I-%s" % label
            if label is not None:
                _add_labeled_span(label_start_char,
                                  len(characterwise_predicted_label_names) - 1,
                                  label, words, proto_labeled_spans)
        output_file.write(documents.SerializeToString())


def _get_main_label_from_bio_label(label_name):
    if _is_label_type(label_name, LabelType.OUTSIDE):
        return LABEL_OUTSIDE
    else:
        assert len(label_name) > 2
        return label_name[2:]  # Strip "B-" or "I-"


def _save_predictions_as_tfrecord(
        output_directory, file_name,
        characterwise_predicted_label_names_per_sentence, words_per_sentence,
        moving_window_overlap, max_seq_length, tokenizer):
    """Saves the hypotheses to an .lftxt file."""
    examples = []
    sentence_id = 0
    example = tagging_data_lib.InputExample(sentence_id=0)
    for (characterwise_predicted_label_names,
         words) in zip(characterwise_predicted_label_names_per_sentence,
                       words_per_sentence):
        selection_start = 0
        selection_label = None
        for i, label_name in enumerate(characterwise_predicted_label_names):
            if i == 0:
                selection_label = _get_main_label_from_bio_label(label_name)
                continue

            # Check if a new label begins.
            if _is_label_type(label_name, LabelType.BEGINNING) or (
                    selection_label != LABEL_OUTSIDE
                    and _is_label_type(label_name, LabelType.OUTSIDE)):
                selection = _get_text_from_character_indices(
                    words, selection_start, i - 1)
                add_tfrecord_label(selection,
                                   selection_label,
                                   tokenizer,
                                   example,
                                   use_additional_labels=True)
                selection_label = _get_main_label_from_bio_label(label_name)
                selection_start = i
        selection = _get_text_from_character_indices(words,
                                                     selection_start,
                                                     end=None)
        add_tfrecord_label(selection,
                           selection_label,
                           tokenizer,
                           example,
                           use_additional_labels=True)
        assert example.words
        examples.append(example)
        sentence_id += 1
        example = tagging_data_lib.InputExample(sentence_id=sentence_id)
    write_example_to_file(examples=examples,
                          tokenizer=tokenizer,
                          max_seq_length=max_seq_length,
                          output_file=os.path.join(output_directory,
                                                   file_name),
                          text_preprocessing=tokenization.convert_to_unicode,
                          moving_window_overlap=moving_window_overlap,
                          mask_overlap=False)


def _filter_unlabeled_sentences(
        characterwise_predicted_label_names_per_sentence, words_per_sentence,
        unlabeled_sentence_filter):
    """Filters sentences without any predicted labels. Keeps every nth entry."""
    sentences_without_label = 0
    filtered_characterwise_predicted_label_names_per_sentence = []
    filtered_words_per_sentence = []
    for (characterwise_predicted_label_names,
         words) in zip(characterwise_predicted_label_names_per_sentence,
                       words_per_sentence):
        if not any([
                _is_label_type(label_name, LabelType.BEGINNING)
                for label_name in characterwise_predicted_label_names
        ]):
            sentences_without_label += 1
            if sentences_without_label % unlabeled_sentence_filter != 0:
                continue
        filtered_characterwise_predicted_label_names_per_sentence.append(
            characterwise_predicted_label_names)
        filtered_words_per_sentence.append(words)
    return (filtered_characterwise_predicted_label_names_per_sentence,
            filtered_words_per_sentence)


def _save_hypotheses(test_name,
                     characterwise_predicted_label_names_per_sentence,
                     words_per_sentence, output_directory,
                     unlabeled_sentence_filters, tokenizer,
                     save_output_formats, moving_window_overlap,
                     max_seq_length):
    """Saves the hypotheses in the specified file formats."""
    if not output_directory:
        raise ValueError(
            "If the hypotheses are supposed to be saved, an output"
            " directory must be specified.")
    for unlabeled_sentence_filter in unlabeled_sentence_filters:
        (filtered_characterwise_predicted_label_names_per_sentence,
         filtered_words_per_sentence) = _filter_unlabeled_sentences(
             characterwise_predicted_label_names_per_sentence,
             words_per_sentence, unlabeled_sentence_filter)
        if unlabeled_sentence_filter == 1:
            file_name_without_extension = test_name
        else:
            file_name_without_extension = "%s_filter_%d" % (
                test_name, unlabeled_sentence_filter)
        if "lftxt" in save_output_formats:
            _save_predictions_as_lftxt(
                output_directory=output_directory,
                file_name="%s.lftxt" % file_name_without_extension,
                characterwise_predicted_label_names_per_sentence=
                filtered_characterwise_predicted_label_names_per_sentence,
                words_per_sentence=filtered_words_per_sentence)
        if "binproto" in save_output_formats:
            _save_predictions_as_binproto(
                output_directory=output_directory,
                file_name="%s.binproto" % file_name_without_extension,
                characterwise_predicted_label_names_per_sentence=
                filtered_characterwise_predicted_label_names_per_sentence,
                words_per_sentence=filtered_words_per_sentence)
        if "tfrecord" in save_output_formats:
            _save_predictions_as_tfrecord(
                output_directory=output_directory,
                file_name="%s.tfrecord" % file_name_without_extension,
                characterwise_predicted_label_names_per_sentence=
                filtered_characterwise_predicted_label_names_per_sentence,
                words_per_sentence=filtered_words_per_sentence,
                moving_window_overlap=moving_window_overlap,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer)


def main(_):
    if len(FLAGS.input_paths) != len(FLAGS.raw_paths):
        raise ValueError("The number of inputs and raw paths must be equal.")

    if FLAGS.tpu_address is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=FLAGS.tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        tokenizer = create_tokenizer_from_hub_module(FLAGS.module_url)
        model, task = _get_model_and_task(FLAGS.module_url, FLAGS.model_path,
                                          FLAGS.train_with_additional_labels)

        for input_path, raw_path in zip(FLAGS.input_paths, FLAGS.raw_paths):
            test_name = os.path.splitext(os.path.basename(raw_path))[0]

            file_reader = get_file_reader(raw_path)

            words_per_sentence = file_reader.get_words(tokenizer)

            characterwise_predicted_label_names_per_sentence = (
                _infer_characterwise_label_names(
                    model, task, input_path,
                    FLAGS.train_with_additional_labels, words_per_sentence,
                    raw_path, tokenizer, FLAGS.batch_size))

            if len(FLAGS.save_output_formats) != 0:
                _save_hypotheses(
                    test_name=test_name,
                    characterwise_predicted_label_names_per_sentence=
                    characterwise_predicted_label_names_per_sentence,
                    words_per_sentence=words_per_sentence,
                    output_directory=FLAGS.output_directory,
                    unlabeled_sentence_filters=FLAGS.
                    unlabeled_sentence_filters,
                    tokenizer=tokenizer,
                    save_output_formats=FLAGS.save_output_formats,
                    moving_window_overlap=FLAGS.moving_window_overlap,
                    max_seq_length=FLAGS.max_seq_length)

            (characterwise_target_labels_per_sentence, characters_per_sentence
             ) = file_reader.get_characterwise_target_labels(tokenizer)

            if FLAGS.visualisation_folder:
                for visualised_label in MAIN_LABELS:
                    _visualise(
                        test_name, characterwise_target_labels_per_sentence,
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

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
"""Readers for different file formats."""

from abc import ABCMeta, abstractmethod
from official.nlp.data import tagging_data_lib
from training.utils import (LABEL_CONTAINER_NAME, LABEL_ID_MAP, LABEL_OUTSIDE,
                            LF_ADDRESS_LABEL, LF_TELEPHONE_LABEL, MAIN_LABELS,
                            MAIN_LABEL_ADDRESS, MAIN_LABEL_TELEPHONE,
                            LabeledExample, add_tfrecord_label,
                            split_into_words, remove_whitespace_and_parse)
from google.protobuf.internal.decoder import _DecodeVarint32

import protocol_buffer.document_pb2 as proto_document

_MAX_BINPROTO_PREFIX_LENGTH = 10


def get_file_reader(path):
    if path.endswith(".binproto"):
        return BinProtoReader(path)
    elif path.endswith(".lftxt"):
        return LftxtReader(path)
    elif path.endswith(".txt"):
        return TxtReader(path)
    else:
        raise NotImplementedError(
            "Only .binproto, .lftxt and .txt input files are supported.")


class FileReader(metaclass=ABCMeta):
    """Base class for file readers."""
    @abstractmethod
    def __init__(self, path):
        pass

    @abstractmethod
    def get_words(self, path, tokenizer):
        pass

    @abstractmethod
    def get_characterwise_target_labels(self, path, tokenizer):
        pass

    @abstractmethod
    def get_examples(self, path, tokenizer, use_additional_labels,
                     use_gold_tokenization_and_include_target_labels):
        pass

    def _update_characterwise_target_labels(self, tokenizer, labeled_example,
                                            characterwise_target_labels):
        """Updates target_labels and characters."""
        prefix_without_whitespace = remove_whitespace_and_parse(
            labeled_example.prefix, tokenizer)
        labeled_text_without_whitespace = remove_whitespace_and_parse(
            labeled_example.selection, tokenizer)
        if len(labeled_text_without_whitespace) > 0:
            start = len(prefix_without_whitespace)
            end = start + len(labeled_text_without_whitespace) - 1
            characterwise_target_labels[start] = "B-%s" % labeled_example.label
            characterwise_target_labels[start + 1:end +
                                        1] = ["I-%s" % labeled_example.label
                                              ] * (end - start)


class BinProtoReader(FileReader):
    """File reader for .binproto files."""
    def __init__(self, path):
        #pylint: disable=super-init-not-called)
        self.path = path

    def get_words(self, tokenizer):
        """Returns all words as defined by the tokenizer."""
        words_per_sentence = []
        for document in self._get_documents():
            words = split_into_words(document.text, tokenizer)
            words_per_sentence.append(words)
        return words_per_sentence

    def get_characterwise_target_labels(self, tokenizer):
        """Returns a label for each character."""
        characterwise_target_labels_per_sentence = []
        characters_per_sentence = []
        for document in self._get_documents():
            characters = remove_whitespace_and_parse(document.text, tokenizer)
            characterwise_target_labels = [LABEL_OUTSIDE] * len(characters)
            total_prefix = ""
            for labeled_example in self._get_labeled_text(
                    document, only_main_labels=True):
                assert labeled_example.suffix == ""
                total_prefix += labeled_example.prefix
                labeled_example = labeled_example._replace(prefix=total_prefix)
                self._update_characterwise_target_labels(
                    tokenizer, labeled_example, characterwise_target_labels)
                total_prefix += labeled_example.selection

            characterwise_target_labels_per_sentence.append(
                characterwise_target_labels)
            characters_per_sentence.append(characters)
        return characterwise_target_labels_per_sentence, characters_per_sentence

    def get_examples(self, tokenizer, use_additional_labels,
                     use_gold_tokenization_and_include_target_labels):
        """Reads one file and returns a list of `InputExample` instances."""
        examples = []
        sentence_id = 0
        example = tagging_data_lib.InputExample(sentence_id=0)
        for document in self._get_documents():
            text = document.text

            if use_gold_tokenization_and_include_target_labels:
                for labeled_example in self._get_labeled_text(document):
                    assert labeled_example.suffix == ""
                    add_tfrecord_label(labeled_example.prefix, LABEL_OUTSIDE,
                                       tokenizer, example,
                                       use_additional_labels)
                    add_tfrecord_label(labeled_example.selection,
                                       labeled_example.label, tokenizer,
                                       example, use_additional_labels)
            else:
                # The tokenizer will split the text without taking the target
                # labels into account.
                add_tfrecord_label(text, LABEL_OUTSIDE, tokenizer, example,
                                   use_additional_labels)

            if example.words:
                examples.append(example)
                sentence_id += 1
                example = tagging_data_lib.InputExample(
                    sentence_id=sentence_id)
        return examples

    def _get_documents(self):
        """Provides an iterator over all documents."""
        document = proto_document.Document()
        with open(self.path, "rb") as src_file:
            msg_buf = src_file.read(_MAX_BINPROTO_PREFIX_LENGTH)
            while msg_buf:
                # Get the message length.
                msg_len, new_pos = _DecodeVarint32(msg_buf, 1)
                msg_buf = msg_buf[new_pos:]
                # Read the rest of the message.
                msg_buf += src_file.read(msg_len - len(msg_buf))
                document.ParseFromString(msg_buf)
                msg_buf = msg_buf[msg_len:]
                # Read the length prefix for the next message.
                msg_buf += src_file.read(_MAX_BINPROTO_PREFIX_LENGTH)
                yield self._convert_token_boundaries_to_codeunits(document)

    def _get_labeled_text(self, document, only_main_labels=False):
        """Provides an iterator over all labeled texts in the proto document."""
        text = document.text

        last_label_end = -1
        for label in document.labeled_spans[LABEL_CONTAINER_NAME].labeled_span:
            if only_main_labels and label.label not in MAIN_LABELS:
                continue

            label_start = document.token[label.token_start].start
            label_end = document.token[label.token_end].end

            prefix = text[last_label_end + 1:label_start]
            labeled_text = text[label_start:label_end + 1]

            last_label_end = label_end

            yield LabeledExample(prefix=prefix,
                                 selection=labeled_text,
                                 suffix="",
                                 complete_text=None,
                                 label=label.label)

        remaining_text = text[last_label_end + 1:]
        yield LabeledExample(prefix=remaining_text,
                             selection="",
                             suffix="",
                             complete_text=None,
                             label=LABEL_OUTSIDE)

    def _convert_token_boundaries_to_codeunits(self, document):
        """Converts the token boundaries from codepoints to codeunits."""
        text_as_string = document.text
        text_as_bytes = bytes(text_as_string, "utf-8")
        for token in document.token:
            prefix_as_bytes = text_as_bytes[:token.start]
            token_as_bytes = text_as_bytes[token.start:token.end + 1]
            prefix_as_string = prefix_as_bytes.decode("utf-8")
            token_as_string = token_as_bytes.decode("utf-8")
            token.start = len(prefix_as_string)
            token.end = len(prefix_as_string) + len(token_as_string) - 1
        return document


class LftxtReader(FileReader):
    """File reader for .lftxt files."""
    def __init__(self, path):
        #pylint: disable=super-init-not-called)
        self.path = path

    def get_words(self, tokenizer):
        """Returns all words as defined by the tokenizer."""
        words_per_sentence = []
        prev_text = ""
        for labeled_example in self.get_labeled_text():
            if prev_text == labeled_example.complete_text:
                continue
            words = split_into_words(labeled_example.complete_text, tokenizer)
            prev_text = labeled_example.complete_text
            words_per_sentence.append(words)
        return words_per_sentence

    def get_characterwise_target_labels(self, tokenizer):
        """Returns a label for each character."""
        characterwise_target_labels_per_sentence = []
        characters_per_sentence = []
        characterwise_target_labels = []
        characters = []
        prev_text = ""
        for labeled_example in self.get_labeled_text():
            if prev_text == labeled_example.complete_text:
                # The last entry is updated, it will be added again.
                del characterwise_target_labels_per_sentence[-1]
                del characters_per_sentence[-1]
            else:
                characters = remove_whitespace_and_parse(
                    labeled_example.complete_text, tokenizer)
                characterwise_target_labels = [LABEL_OUTSIDE] * len(characters)

            self._update_characterwise_target_labels(
                tokenizer, labeled_example, characterwise_target_labels)

            characterwise_target_labels_per_sentence.append(
                characterwise_target_labels)
            characters_per_sentence.append(characters)
            prev_text = labeled_example.complete_text

        return characterwise_target_labels_per_sentence, characters_per_sentence

    def get_examples(self, tokenizer, use_additional_labels,
                     use_gold_tokenization_and_include_target_labels):
        """Reads one file and returns a list of `InputExample` instances."""
        examples = []
        sentence_id = 0
        example = tagging_data_lib.InputExample(sentence_id=0)
        prev_text = ""
        for labeled_example in self.get_labeled_text():
            if use_gold_tokenization_and_include_target_labels:
                if prev_text == labeled_example.complete_text:
                    # Recover the previous example object.
                    sentence_id -= 1
                    example = examples[-1]
                    del examples[-1]
                    prefix_word_length = len(
                        split_into_words(labeled_example.prefix, tokenizer))
                    if any([
                            label_id != LABEL_ID_MAP[LABEL_OUTSIDE] for
                            label_id in example.label_ids[prefix_word_length:]
                    ]):
                        raise NotImplementedError(
                            "If the .lftxt file contains the same sentence"
                            " multiple times, they are assumed to be sorted in"
                            " the order of labelled sequences.")
                    del example.label_ids[prefix_word_length:]
                    del example.words[prefix_word_length:]
                else:
                    add_tfrecord_label(labeled_example.prefix, LABEL_OUTSIDE,
                                       tokenizer, example,
                                       use_additional_labels)
                add_tfrecord_label(labeled_example.selection,
                                   labeled_example.label, tokenizer, example,
                                   use_additional_labels)
                add_tfrecord_label(labeled_example.suffix, LABEL_OUTSIDE,
                                   tokenizer, example, use_additional_labels)
            else:
                if prev_text == labeled_example.complete_text:
                    continue
                add_tfrecord_label(labeled_example.complete_text,
                                   LABEL_OUTSIDE,
                                   tokenizer,
                                   example,
                                   use_additional_labels=False)
            prev_text = labeled_example.complete_text

            if example.words:
                examples.append(example)
                sentence_id += 1
                example = tagging_data_lib.InputExample(
                    sentence_id=sentence_id)
        return examples

    def get_labeled_text(self):
        """Provides an iterator over all labeled texts in the linkfragments."""
        with open(self.path, "r") as file:
            for linkfragment in file:
                text, label_description = linkfragment.split("\t")
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

                text_without_braces = text.replace("{{{",
                                                   "").replace("}}}", "")

                yield LabeledExample(prefix=prefix,
                                     selection=labeled_text,
                                     suffix=suffix,
                                     complete_text=text_without_braces,
                                     label=label)


class TxtReader(FileReader):
    """File reader for .txt files."""
    def __init__(self, path):
        #pylint: disable=super-init-not-called)
        self.path = path

    def get_words(self, tokenizer):
        """Returns all words as defined by the tokenizer."""
        words_per_sentence = []
        prev_text_without_braces = ""
        with open(self.path, "r") as file:
            for text_without_braces in file:
                if prev_text_without_braces == text_without_braces:
                    continue
                prev_text_without_braces = text_without_braces
                words = split_into_words(text_without_braces, tokenizer)
                words_per_sentence.append(words)
        return words_per_sentence

    def get_characterwise_target_labels(self, tokenizer):
        """Returns the outside label for each character"""
        characterwise_target_labels_per_sentence = []
        characters_per_sentence = []
        characterwise_target_labels = []
        characters = []
        prev_text = ""
        with open(self.path, "r") as file:
            for text in file:
                if prev_text == text:
                    continue
                else:
                    characters = remove_whitespace_and_parse(text, tokenizer)
                    characterwise_target_labels = [LABEL_OUTSIDE
                                                   ] * len(characters)

                characterwise_target_labels_per_sentence.append(
                    characterwise_target_labels)
                characters_per_sentence.append(characters)
                prev_text = text

        return characterwise_target_labels_per_sentence, characters_per_sentence

    def get_examples(self, tokenizer, use_additional_labels,
                     use_gold_tokenization_and_include_target_labels):
        """Reads one file and returns a list of `InputExample` instances."""
        if use_gold_tokenization_and_include_target_labels:
            raise ValueError(
                ".txt file contain no labeling information, they can only be"
                " used for inference.")
        examples = []
        sentence_id = 0
        example = tagging_data_lib.InputExample(sentence_id=0)
        prev_text = ""
        with open(self.path, "r") as file:
            for text in file:
                if prev_text == text:
                    continue
                add_tfrecord_label(text,
                                   LABEL_OUTSIDE,
                                   tokenizer,
                                   example,
                                   use_additional_labels=False)
                prev_text = text

                if example.words:
                    examples.append(example)
                    sentence_id += 1
                    example = tagging_data_lib.InputExample(
                        sentence_id=sentence_id)
        return examples

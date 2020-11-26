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
"""Automatic tests to ensure training/evaluation succeed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from absl import flags
from absl.testing import absltest
from tensorflow.python.lib.io.file_io import filecmp

FLAGS = flags.FLAGS

_train_textproto_content = """documents {
  text: "At 42 Street or here."
  token: {
    word: "At"
    start: 0
    end: 1
  }
  token: {
    word: "42"
    start: 3
    end: 4
  }
  token: {
    word: "Street"
    start: 6
    end: 11
  }
  token: {
    word: "or"
    start: 13
    end: 14
  }
  token: {
    word: "here"
    start: 16
    end: 19
  }
  token: {
    word: "."
    start: 20
    end: 20
  }
  labeled_spans: {
    key: "lucid"
    value: {
      labeled_span: {
        token_start: 1
        token_end: 2
        label: "ADDRESS"
      }
      labeled_span: {
        token_start: 4
        token_end: 4
        label: "ADDRESS"
      }
    }
  }
}"""


def _get_dependency(name):
    """Returns the executable path of a dependency.
    Args:
        binary_name: string, the name of a dependency.
    Raises:
        FileNotFoundError: Raised when it cannot locate the dependency path.
    """
    # Get the base path
    path = os.path.dirname(os.path.dirname(__file__))
    # Return the first matching file
    for subdir, _, files in os.walk(path):
        for file in files:
            if file == name:
                return os.path.join(subdir, file)

    raise FileNotFoundError("Binary %s not found" % name)


class IntegrationTests(absltest.TestCase):
    """End to end integration tests to ensure all scripts can be run in
    sequence."""
    def run_helper(self,
                   program_name,
                   expect_success=True,
                   expected_stdout_substrings=(),
                   expected_stderr_substrings=(),
                   arguments=(),
                   env_overrides=None):
        """Executes the given script, asserting the defined behaviour.

        Slightly modified version of FunctionalTests.run_helper in
        https://github.com/abseil/abseil-py/blob/m
        aster/absl/tests/app_test.py
        """
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf8"
        if env_overrides:
            env.update(env_overrides)

        process = subprocess.Popen([_get_dependency(program_name)] +
                                   list(arguments),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=env,
                                   universal_newlines=True)
        stdout, stderr = process.communicate()

        print(program_name, stdout)
        print(program_name, stderr)

        message = (u"Command: {command}\n"
                   "Exit Code: {exitcode}\n"
                   "===== stdout =====\n{stdout}"
                   "===== stderr =====\n{stderr}"
                   "==================".format(
                       command=" ".join([program_name] + list(arguments)),
                       exitcode=process.returncode,
                       stdout=stdout or "<no output>\n",
                       stderr=stderr or "<no output>\n"))
        if expect_success:
            self.assertEqual(0, process.returncode, msg=message)
        else:
            self.assertNotEqual(0, process.returncode, msg=message)

        for expected_stdout_substring in expected_stdout_substrings:
            self.assertIn(expected_stdout_substring, stdout, message)
        for expected_stderr_substring in expected_stderr_substrings:
            self.assertIn(expected_stderr_substring, stderr, message)

    def setUp(self):
        """Creates temporary files for training/evaluation."""
        self.out_dir = self.create_tempdir()
        self.augmenter_replacement_input = os.path.join(
            self.out_dir, "replacements.txt")
        with open(self.augmenter_replacement_input, "w") as file:
            file.write("Replacement\t1")

        self.train_data_dir = os.path.join(self.out_dir, "train")
        os.makedirs(self.train_data_dir)
        self.train_textproto = os.path.join(self.train_data_dir,
                                            "train.textproto")
        with open(self.train_textproto, "w") as f:
            f.write(_train_textproto_content)
        training_text = [
            "At {{{42 Street}}} or here.\taddress\n",
            "At 42 Street or {{{here}}}.\taddress"
        ]
        self.train_lftxt = os.path.join(self.train_data_dir, "train.lftxt")
        with open(self.train_lftxt, "w") as f:
            f.writelines(training_text)

        # test on train corpus to see overfitting.
        self.test_data_dir = os.path.join(self.out_dir, "test")
        os.makedirs(self.test_data_dir)
        self.test_textproto = os.path.join(self.test_data_dir,
                                           "test.textproto")
        with open(self.test_textproto, "w") as f:
            f.write(_train_textproto_content)
        self.test_lftxt = os.path.join(self.test_data_dir, "test.lftxt")
        with open(self.test_lftxt, "w") as f:
            f.writelines(training_text)

        self.test2_data_dir = os.path.join(self.out_dir, "test2")
        os.makedirs(self.test2_data_dir)
        self.test2_lftxt = os.path.join(self.test2_data_dir, "test2.lftxt")
        with open(self.test2_lftxt, "w") as f:
            f.writelines([
                "Not a {{{real address}}}.\taddress\n",
                "Phone number: {{{00 - 11 222 333}}} and 911!\tphone\n",
                "Phone number: 00 - 11 222 333 and {{{911}}}!\tphone"
            ])

        self.test3_data_dir = os.path.join(self.out_dir, "test3")
        os.makedirs(self.test3_data_dir)
        self.test3_txt = os.path.join(self.test3_data_dir, "test3.txt")
        with open(self.test3_txt, "w") as f:
            f.writelines([
                "Not a real address.\n",
                "Phone number: 00 - 11 222 333 and 911!"
            ])

        self.train_tfrecord = os.path.join(self.train_data_dir,
                                           "train.tfrecord")
        self.dev_tfrecord = os.path.join(self.train_data_dir, "dev.tfrecord")
        self.test_tfrecord = os.path.join(self.test_data_dir, "test.tfrecord")
        self.test2_tfrecord = os.path.join(self.test2_data_dir,
                                           "test2.tfrecord")
        self.test3_tfrecord = os.path.join(self.test3_data_dir,
                                           "test3.tfrecord")
        self.meta_data = os.path.join(self.train_data_dir, "meta.data")
        self.module_url = (
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"  # pylint: disable=line-too-long
        )
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")

    def test_training(self):
        """Test normal training."""
        self.run_helper(
            "main",
            arguments=("--corpora", "train", "--input_directory",
                       self.train_data_dir, "--output_directory",
                       self.train_data_dir, "--addresses_path",
                       self.augmenter_replacement_input, "--phones_path",
                       self.augmenter_replacement_input, "--num_total", "0"))
        train_binproto = os.path.join(self.train_data_dir, "train.binproto")
        self.run_helper(
            "convert_data",
            arguments=("--module_url", self.module_url,
                       "--train_data_input_path", train_binproto,
                       "--train_data_output_path", self.train_tfrecord,
                       "--dev_data_input_path", train_binproto,
                       "--dev_data_output_path", self.dev_tfrecord,
                       "--test_data_input_paths", train_binproto,
                       "--test_data_output_paths", self.test_tfrecord,
                       "--test_data_input_paths", self.test2_lftxt,
                       "--test_data_output_paths", self.test2_tfrecord,
                       "--meta_data_file_path", self.meta_data))
        self.run_helper("train",
                        arguments=("--module_url", self.module_url,
                                   "--train_data_path", self.train_tfrecord,
                                   "--validation_data_path", self.dev_tfrecord,
                                   "--epochs", "1", "--train_size", "128",
                                   "--save_path", self.checkpoint_dir))
        model_path = os.path.join(self.checkpoint_dir, "model_01")
        visualisation_dir = os.path.join(self.out_dir, "visualisation")
        self.run_helper(
            "evaluate",
            arguments=("--module_url", self.module_url, "--model_path",
                       model_path, "--input_paths", self.train_tfrecord,
                       "--raw_paths", train_binproto, "--input_paths",
                       self.test_tfrecord, "--raw_paths", self.test_lftxt,
                       "--input_paths", self.test2_tfrecord, "--raw_paths",
                       self.test2_lftxt, "--visualisation_folder",
                       visualisation_dir))

    def test_training_additional_labels(self):
        """Training with additional labels."""
        self.run_helper(
            "convert_data",
            arguments=("--module_url", self.module_url,
                       "--train_data_input_path", self.train_lftxt,
                       "--train_data_output_path", self.train_tfrecord,
                       "--dev_data_input_path", self.train_lftxt,
                       "--dev_data_output_path", self.dev_tfrecord,
                       "--test_data_input_paths", self.test_lftxt,
                       "--test_data_output_paths", self.test_tfrecord,
                       "--test_data_input_paths", self.test2_lftxt,
                       "--test_data_output_paths", self.test2_tfrecord,
                       "--meta_data_file_path", self.meta_data,
                       "--train_with_additional_labels"))
        self.run_helper("train",
                        arguments=("--module_url", self.module_url,
                                   "--train_data_path", self.train_tfrecord,
                                   "--validation_data_path", self.dev_tfrecord,
                                   "--epochs", "1", "--train_size", "128",
                                   "--save_path", self.checkpoint_dir,
                                   "--train_with_additional_labels"))
        model_path = os.path.join(self.checkpoint_dir, "model_01")
        visualisation_dir = os.path.join(self.out_dir, "visualisation")
        self.run_helper(
            "evaluate",
            arguments=("--module_url", self.module_url, "--model_path",
                       model_path, "--input_paths", self.test_tfrecord,
                       "--raw_paths", self.test_lftxt, "--input_paths",
                       self.test2_tfrecord, "--raw_paths", self.test2_lftxt,
                       "--visualisation_folder", visualisation_dir,
                       "--train_with_additional_labels"))

    def test_training_last_layer_only(self):
        """Training of the last layer only, the bert model is frozen."""
        self.run_helper(
            "convert_data",
            arguments=("--module_url", self.module_url,
                       "--train_data_input_path", self.train_lftxt,
                       "--train_data_output_path", self.train_tfrecord,
                       "--dev_data_input_path", self.train_lftxt,
                       "--dev_data_output_path", self.dev_tfrecord,
                       "--test_data_input_paths", self.test_lftxt,
                       "--test_data_output_paths", self.test_tfrecord,
                       "--test_data_input_paths", self.test2_lftxt,
                       "--test_data_output_paths", self.test2_tfrecord,
                       "--meta_data_file_path", self.meta_data))
        self.run_helper("train",
                        arguments=("--module_url", self.module_url,
                                   "--train_data_path", self.train_tfrecord,
                                   "--validation_data_path", self.dev_tfrecord,
                                   "--epochs", "1", "--train_size", "128",
                                   "--save_path", self.checkpoint_dir,
                                   "--train_last_layer_only"))
        model_path = os.path.join(self.checkpoint_dir, "model_01")
        visualisation_dir = os.path.join(self.out_dir, "visualisation")
        self.run_helper(
            "evaluate",
            arguments=("--module_url", self.module_url, "--model_path",
                       model_path, "--input_paths", self.test_tfrecord,
                       "--raw_paths", self.test_lftxt, "--input_paths",
                       self.test2_tfrecord, "--raw_paths", self.test2_lftxt,
                       "--visualisation_folder", visualisation_dir))

    def test_evaluate_lftxt(self):
        """Evaluate precomputed lftxt files."""
        visualisation_dir = os.path.join(self.out_dir, "visualisation")
        self.run_helper(
            "evaluate",
            arguments=("--module_url", self.module_url, "--input_paths",
                       self.test_data_dir, "--raw_paths", self.test_lftxt,
                       "--visualisation_folder", visualisation_dir))

    def test_file_format_equivalence(self):
        """Test data conversion."""
        self.run_helper(
            "main",
            arguments=("--corpora", "train", "--input_directory",
                       self.train_data_dir, "--output_directory",
                       self.train_data_dir, "--addresses_path",
                       self.augmenter_replacement_input, "--phones_path",
                       self.augmenter_replacement_input, "--num_total", "0"))
        train_binproto = os.path.join(self.train_data_dir, "train.binproto")
        self.run_helper(
            "convert_data",
            arguments=("--module_url", self.module_url,
                       "--train_data_input_path", train_binproto,
                       "--train_data_output_path", self.train_tfrecord,
                       "--dev_data_input_path", self.test_lftxt,
                       "--dev_data_output_path", self.dev_tfrecord,
                       "--test_data_input_paths", self.test2_lftxt,
                       "--test_data_output_paths", self.test2_tfrecord,
                       "--test_data_input_paths", self.test3_txt,
                       "--test_data_output_paths", self.test3_tfrecord,
                       "--meta_data_file_path", self.meta_data))
        filecmp(self.train_tfrecord, self.dev_tfrecord)
        filecmp(self.test2_tfrecord, self.test3_tfrecord)


if __name__ == "__main__":
    absltest.main()

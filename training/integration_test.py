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
import re
import subprocess

from absl import flags
from absl.testing import absltest

FLAGS = flags.FLAGS


def _normalize_newlines(s):
    return re.sub("(\r\n)|\r", "\n", s)


def _get_executable_path(binary_name):
    """Returns the executable path of a binary.
    Args:
        binary_name: string, the name of a binary.
    Raises:
        RuntimeError: Raised when it cannot locate the executable path.
    """
    # Get the base path
    path = os.path.dirname(os.path.dirname(__file__))
    # Return the first matching file
    for subdir, _, files in os.walk(path):
        for file in files:
            if file == binary_name:
                return os.path.join(subdir, file)

    raise RuntimeError("Binary %s not found" % binary_name)


class IntegrationTests(absltest.TestCase):
    """End to end integration tests to ensure all scripts can be run in
    sequence."""
    def run_helper(self,
                   program_name,
                   expect_success=True,
                   expected_stdout_substrings=None,
                   expected_stderr_substrings=None,
                   arguments=(),
                   env_overrides=None):
        """Executes the given script, asserting the defined behaviour.

        Slightly modified version of FunctionalTests.run_helper in
        https://github.com/abseil/abseil-py/blob/m
        aster/absl/tests/app_test.py
        """
        if expected_stdout_substrings is None:
            expected_stdout_substrings = []
        if expected_stderr_substrings is None:
            expected_stderr_substrings = []
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf8"
        if env_overrides:
            env.update(env_overrides)

        process = subprocess.Popen([_get_executable_path(program_name)] +
                                   list(arguments),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=env,
                                   universal_newlines=False)
        stdout, stderr = process.communicate()
        # In Python 2, we can"t control the encoding used by universal_newline
        # mode, which can cause UnicodeDecodeErrors when subprocess tries to
        # conver the bytes to unicode, so we have to decode it manually.
        stdout = _normalize_newlines(stdout.decode("utf8"))
        stderr = _normalize_newlines(stderr.decode("utf8"))

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
        training_text = [
            "Meet at {{{221b Baker Street}}}.\taddress\n",
            "Call at {{{+01 2345 6789}}}!\tphone"
        ]
        self.train_lftxt = os.path.join(self.out_dir, "train.lftxt")
        with open(self.train_lftxt, "w") as f:
            f.writelines(training_text)
        # test on train corpus to see overfitting.
        self.test_lftxt = os.path.join(self.out_dir, "test.lftxt")
        with open(self.test_lftxt, "w") as f:
            f.writelines(training_text)
        self.test2_lftxt = os.path.join(self.out_dir, "test2.lftxt")
        with open(self.test2_lftxt, "w") as f:
            f.writelines([
                "Not a {{{real address}}}.\taddress\n",
                "Phone number: {{{00 - 11 222 333}}}!\tphone"
            ])

        self.train_tfrecord = os.path.join(self.out_dir, "train.tfrecord")
        self.dev_tfrecord = os.path.join(self.out_dir, "dev.tfrecord")
        self.test_tfrecord = os.path.join(self.out_dir, "test.tfrecord")
        self.test2_tfrecord = os.path.join(self.out_dir, "test2.tfrecord")
        self.meta_data = os.path.join(self.out_dir, "meta.data")
        self.module_url = (
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"  # pylint: disable=line-too-long
        )

    def test_training(self):
        """Normal training."""

        checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
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
                                   "--epochs", "1", "--train_size", "6400",
                                   "--save_path", checkpoint_dir))
        model_path = os.path.join(checkpoint_dir, "model_01")
        self.run_helper("evaluate",
                        arguments=("--module_url", self.module_url,
                                   "--model_path", model_path,
                                   "--test_data_paths", self.test_tfrecord,
                                   "--test_data_paths", self.test2_tfrecord))


if __name__ == "__main__":
    absltest.main()

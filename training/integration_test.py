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


def _normalize_newlines(s):
    return re.sub("(\r\n)|\r", "\n", s)


FLAGS = flags.FLAGS


def get_executable_path(py_binary_name):
    """Returns the executable path of a py_binary.
  This returns the executable path of a py_binary that is in another Bazel
  target"s data dependencies.
  On Linux/macOS, the path and __file__ has the same root directory.
  On Windows, bazel builds an .exe file and we need to use the MANIFEST file
  the location the actual binary.
  Args:
    py_binary_name: string, the name of a py_binary that is in another Bazel
        target"s data dependencies.
  Raises:
    RuntimeError: Raised when it cannot locate the executable path.
  """
    if os.name == "nt":
        py_binary_name += ".exe"
        manifest_file = os.path.join(FLAGS.test_srcdir, "MANIFEST")
        workspace_name = os.environ["TEST_WORKSPACE"]
        manifest_entry = "{}/{}".format(workspace_name, py_binary_name)
        with open(manifest_file, "r") as manifest_fd:
            for line in manifest_fd:
                tokens = line.strip().split(" ")
                if len(tokens) != 2:
                    continue
                if manifest_entry == tokens[0]:
                    return tokens[1]
        raise RuntimeError(
            "Cannot locate executable path for {}, MANIFEST file: {}.".format(
                py_binary_name, manifest_file))
    else:
        # NOTE: __file__ may be .py or .pyc, depending on how the module was
        # loaded and executed.
        path = __file__

        # Use the package name to find the root directory: every dot is
        # a directory, plus one for ourselves.
        for _ in range(__name__.count(".") + 1):
            path = os.path.dirname(path)

        root_directory = path
        return os.path.join(root_directory, py_binary_name)


class IntegrationTests(absltest.TestCase):
    """End to end integration tests to ensure all scripts can be run in
    sequence."""
    def run_helper(self,
                   program_name,
                   expect_success=True,
                   expected_stdout_substring=None,
                   expected_stderr_substring=None,
                   arguments=(),
                   env_overrides=None):
        """Executes the given script, asserting the defined behaviour.

        Slightly modified version of FunctionalTests.run_helper in
        https://github.com/abseil/abseil-py/blob/master/absl/tests/app_test.py
        """
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf8"
        if env_overrides:
            env.update(env_overrides)

        process = subprocess.Popen([get_executable_path(program_name)] +
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

        if expected_stdout_substring:
            self.assertIn(expected_stdout_substring, stdout, message)
        if expected_stderr_substring:
            self.assertIn(expected_stderr_substring, stderr, message)

    def setUp(self):
        """Creates temporary files for training/evaluation."""
        out_dir = self.create_tempdir()
        self.input_lftxt = os.path.join(out_dir, "tmp.lftxt")
        with open(self.input_lftxt, "w") as f:
            f.writelines([
                "Meet at {{{221b Baker Street}}}.\taddress\n",
                "Call at {{{+01 2345 6789}}}!\tphone"
            ])

        self.train_tfrecord = os.path.join(out_dir, "train.tfrecord")
        self.dev_tfrecord = os.path.join(out_dir, "dev.tfrecord")
        self.test_tfrecord = os.path.join(out_dir, "test.tfrecord")
        self.meta_data = os.path.join(out_dir, "meta.data")

    def test_training(self):
        """Normal training."""
        self.run_helper(
            "convert_data",
            arguments=
            ("--module_url",
             "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
             "--train_data_input_path", self.input_lftxt,
             "--train_data_output_path", self.train_tfrecord,
             "--dev_data_input_path", self.input_lftxt,
             "--dev_data_output_path", self.dev_tfrecord,
             "--test_data_input_paths", self.input_lftxt,
             "--test_data_output_paths", self.test_tfrecord,
             "--meta_data_file_path", self.meta_data))


if __name__ == "__main__":
    absltest.main()

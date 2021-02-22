#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")
load("@pip_dependencies//:requirements.bzl", "requirement")

py_library(
  name = "utils",
  srcs = ["utils.py"],
  deps = [
    ":model_setup_config",
    "@com_google_research_bert//:tokenization",
    requirement("tf_models_nightly"),
  ],
)

py_library(
  name = "configurable_training_tagging_task",
  srcs = ["configurable_training_tagging_task.py"],
  deps = [
    requirement("tf_models_nightly"),
  ],
)

py_library(
  name = "file_reader",
  srcs = ["file_reader.py"],
  deps = [
    ":utils",
    "//protocol_buffer:documents_py_proto",
    requirement("tf_models_nightly"),
  ],
)

py_library(
  name = "model_setup_config",
  srcs = ["model_setup_config.py"],
)

py_binary(
  name = "convert_data",
  srcs = ["convert_data.py"],
  deps = [
    ":utils",
    ":file_reader",
    ":model_setup_config",
    "@com_google_research_bert//:tokenization",
    requirement("tf_models_nightly"),
  ],
  python_version = "PY3",
)

py_binary(
  name = "train",
  srcs = ["train.py"],
  deps = [
    ":utils",
    ":model_setup_config",
    ":configurable_training_tagging_task",
    requirement("tf_models_nightly"),
  ],
  python_version = "PY3",
)

py_binary(
  name = "evaluate",
  srcs = ["evaluate.py"],
  deps = [
    ":model_setup_config",
    ":utils",
    ":file_reader",
    "//protocol_buffer:documents_py_proto",
    requirement("tf_models_nightly"),
  ],
  python_version = "PY3",
)

py_test (
  name = "integration_test",
  srcs = [
    "integration_test.py"
  ],
  deps = [
    ":convert_data",
    ":train",
    ":evaluate",
    requirement("tf_models_nightly"),
  ],
  data = [
    "//augmenter:main",
  ],
  size = "large",
  python_version = "PY3",
)

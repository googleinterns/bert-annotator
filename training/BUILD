load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")
load("@pip_dependencies//:requirements.bzl", "requirement")

py_library(
  name = "utils",
  srcs = ["utils.py"],
  deps = [
    "//protocol_buffer:documents_py_proto",
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

py_binary(
  name = "convert_data",
  srcs = ["convert_data.py"],
  deps = [
    ":utils",
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
    ":configurable_training_tagging_task",
    requirement("tf_models_nightly"),
  ],
  python_version = "PY3",
)

py_binary(
  name = "evaluate",
  srcs = ["evaluate.py"],
  deps = [
    ":utils",
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

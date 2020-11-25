load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")
load("@pip_dependencies//:requirements.bzl", "requirement")

py_library(
  name = "utils",
  srcs = ["utils.py"],
  deps = [
    "@com_google_research_bert//:tokenization",
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

py_binary(
  name = "convert_data",
  srcs = ["convert_data.py"],
  deps = [
    ":utils",
    ":file_reader",
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
    requirement("tf_models_nightly"),
  ],
  python_version = "PY3",
)

py_binary(
  name = "evaluate",
  srcs = ["evaluate.py"],
  deps = [
    ":utils",
    ":file_reader",
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

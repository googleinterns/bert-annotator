load("@rules_python//python:defs.bzl", "py_library", "py_binary")

py_library(
  name = "utils",
  srcs = ["utils.py"],
)

py_binary(
  name = "convert_data",
  srcs = ["convert_data.py"],
  deps = [
    ":utils",
    "//protocol_buffer:documents_py_proto",
    "@com_google_research_bert//:tokenization",
  ],
)

py_binary(
  name = "train",
  srcs = ["train.py"],
  deps = [
    ":utils",
  ]
)

py_binary(
  name = "evaluate",
  srcs = ["evaluate.py"],
  deps = [
    ":utils",
  ]
)

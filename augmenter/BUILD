load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

cc_library(
  name = "percentage",
  srcs = ["percentage.cc"],
  hdrs = ["percentage.h"],
  deps = [
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
  ]
)

cc_library(
  name = "augmenter",
  srcs = ["augmenter.cc"],
  hdrs = ["augmenter.h"],
  deps = [
    "percentage",
    "//protocol_buffer:documents_cc_proto",
    "@boost//:algorithm",
  ]
)

cc_test (
  name = "augmenter_test",
  srcs = [
    "augmenter_test.cc"
  ],
  deps = [
    "percentage",
    "augmenter",
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/flags:flag",
    "@gtest//:gtest",
    "@gtest//:gtest_main",
  ],
  size = "small"
)

cc_library(
  name = "textproto_io",
  srcs = ["textproto_io.cc"],
  hdrs = ["textproto_io.h"],
  deps = [
    "//protocol_buffer:documents_cc_proto",
  ]
)

cc_binary(
  name = "main",
  srcs = ["main.cc", "main.h"],
  deps = [
    "percentage",
    "augmenter",
    "textproto_io",
    "//protocol_buffer:documents_cc_proto",
    "@boost//:algorithm",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:marshalling",
    "@com_google_absl//absl/flags:parse",
  ],
)

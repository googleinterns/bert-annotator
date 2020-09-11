load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_binary(
  name = "main",
  srcs = ["main.cc", "main.h", "augmenter.cc", "augmenter.h", "percentage.cc", "percentage.h", "textproto_io.cc", "textproto_io.h"],
  deps = [
    "//protocol_buffer:documents_cc_proto",
    "@boost//:algorithm",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:marshalling",
    "@com_google_absl//absl/flags:parse",
  ],
)


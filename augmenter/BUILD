load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_binary(
  name = "augmenter",
  srcs = ["augmenter.cc", "augmenter.h"],
  deps = [
    "//protocol_buffer:documents_cc_proto",
    "@boost//:algorithm",
    "@com_google_absl//absl/flags:flag"
  ],
)


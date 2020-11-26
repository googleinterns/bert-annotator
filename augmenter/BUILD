load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

cc_library(
  name = "case_augmentation",
  hdrs = ["case_augmentation.h"],
)

cc_library(
  name = "token_range",
  hdrs = ["token_range.h"],
)

cc_library(
  name = "augmentations",
  srcs = ["augmentations.cc"],
  hdrs = ["augmentations.h"],
)

cc_library(
  name = "augmenter",
  srcs = ["augmenter.cc"],
  hdrs = ["augmenter.h"],
  deps = [
    ':augmentations',
    ':case_augmentation',
    ":random_sampler",
    ":shuffler",
    ":token_range",
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/random",
    "@com_google_absl//absl/random:bit_gen_ref",
    "@com_google_absl//absl/strings:strings",
  ]
)

cc_test (
  name = "augmenter_test",
  srcs = [
    "augmenter_test.cc"
  ],
  deps = [
    ':augmentations',
    ":augmenter",
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/random:mock_distributions",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
  ],
  size = "small"
)

cc_library(
  name = "proto_io",
  srcs = ["proto_io.cc"],
  hdrs = ["proto_io.h"],
  deps = [
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/strings:strings",
  ]
)

cc_library(
  name = "shuffler",
  srcs = ["shuffler.cc"],
  hdrs = ["shuffler.h"],
  deps = [
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/algorithm:container",
    "@com_google_absl//absl/random:bit_gen_ref",
    "@com_google_googletest//:gtest",
  ]
)

cc_library(
  name = "random_item",
  srcs = ["random_item.cc"],
  hdrs = ["random_item.h"],
)

cc_library(
  name = "random_sampler",
  srcs = ["random_sampler.cc"],
  hdrs = ["random_sampler.h"],
  deps = [
    ":random_item",
    "@com_google_absl//absl/random",
    "@com_google_absl//absl/random:bit_gen_ref",
    "@com_google_googletest//:gtest",
  ],
)

cc_test (
  name = "random_sampler_test",
  srcs = [
    "random_sampler_test.cc"
  ],
  deps = [
    ":random_sampler",
    "@com_google_absl//absl/random:mock_distributions",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
  ],
  size = "small"
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ':augmentations',
    ":augmenter",
    ":random_sampler",
    ":proto_io",
    "//protocol_buffer:documents_cc_proto",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:marshalling",
    "@com_google_absl//absl/flags:parse",
  ],
  visibility = ["//training:__pkg__"],
)

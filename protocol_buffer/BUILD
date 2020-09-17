load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

cc_proto_library(
    name = "documents_cc_proto",
    deps = [":documents_proto"],
    visibility = ["//augmenter:__pkg__"],
)

proto_library(
    name = "documents_proto",
    srcs = ["document.proto", "documents.proto"],
)

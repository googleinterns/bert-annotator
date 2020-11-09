load("@com_google_protobuf//:protobuf.bzl", "py_proto_library",
     "cc_proto_library")

cc_proto_library(
    name = "documents_cc_proto",
    srcs = ["document.proto", "documents.proto"],
    visibility = ["//augmenter:__pkg__"],
)

py_proto_library(
    name = "documents_py_proto",
    srcs = ["document.proto", "documents.proto"],
    visibility = ["//training:__pkg__"],
)

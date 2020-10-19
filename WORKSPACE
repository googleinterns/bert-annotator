load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

# Python rules
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
    strip_prefix = "rules_python-0.0.2",
    sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
)


# Protobuffer
git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf",
    commit = "6d4e7fd7966c989e38024a8ea693db83758944f1",
    shallow_since = "1570061847 -0700",
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Abseil
git_repository(
    name = "com_google_absl",
    commit = "f2c9c663db28a8a898c1fc8c8e06ab9b93eb5610",
    remote = "https://github.com/abseil/abseil-cpp",
    shallow_since = "1599747040 -0400",
)

# GoogleTest
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    shallow_since = "1570114335 -0400",
)

# Bert (for tokenization)
new_git_repository(
    name = "com_google_research_bert",
    remote = "https://github.com/google-research/bert",
    commit = "eedf5716ce1268e56f0a50264a88cafad334ac61",
    shallow_since = "1583939994 -0400",
    build_file = "BUILD.bert",
)

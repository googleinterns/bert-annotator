#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

# Python rules
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
   name = "pip_dependencies",
   requirements = "requirements.txt",
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

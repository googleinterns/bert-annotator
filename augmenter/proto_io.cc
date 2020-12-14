//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "augmenter/proto_io.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

bool ProtoIO::Load(const absl::string_view path) {
  if (absl::EndsWith(path, ".binproto")) {
    return LoadBinary(path);
  } else if (absl::EndsWith(path, ".textproto")) {
    return LoadTextproto(path);
  } else {
    std::cerr << "File format of file " << path << " is not supported"
              << std::endl;
    return false;
  }
}

bool ProtoIO::LoadTextproto(const absl::string_view path) {
  std::ifstream input(std::string(path), std::ios::in);
  if (input.fail()) {
    std::cerr << "Failed to load corpus " << path << std::endl;
    return false;
  }
  google::protobuf::io::IstreamInputStream fileInput(&input, std::ios::binary);
  if (!google::protobuf::TextFormat::Parse(&fileInput, &documents_)) {
    std::cerr << "Failed to parse corpus " << path << std::endl;
    return false;
  }
  return true;
}

bool ProtoIO::LoadBinary(const absl::string_view path) {
  std::ifstream input(std::string(path), std::ios::in | std::ios::binary);
  if (input.fail()) {
    std::cerr << "Failed to load corpus " << path << std::endl;
    return false;
  }
  if (!documents_.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse corpus " << path << std::endl;
    return false;
  }
  return true;
}

bool ProtoIO::Save(const absl::string_view path,
                   const int output_sentences_per_file) const {
  if (output_sentences_per_file == -1) {
    std::cout << "Saving to " << path << std::endl;
    return Save(path, &documents_);
  }

  int file_number = 0;
  for (int start = 0; start < documents_.documents_size();
       start += output_sentences_per_file) {
    int end = start + output_sentences_per_file;
    if (end > documents_.documents_size()) {
      end = documents_.documents_size();
    }
    bert_annotator::Documents documents_to_save;
    *documents_to_save.mutable_documents() = {
        documents_.documents().begin() + start, documents_.documents().begin() + end};
    const std::string shard_path = absl::StrFormat(path, file_number);

    std::cout << "Saving to " << shard_path << std::endl;
    const bool success = Save(shard_path, &documents_to_save);
    if (!success) {
      return false;
    }
    file_number += 1;
  }
  return true;
}

bool ProtoIO::Save(absl::string_view path,
                   const bert_annotator::Documents* documents_to_save) const {
  if (absl::EndsWith(path, ".binproto")) {
    return SaveBinary(path, documents_to_save);
  } else if (absl::EndsWith(path, ".textproto")) {
    return SaveTextproto(path, documents_to_save);
  } else if (absl::EndsWith(path, ".txt")) {
    return SaveTxt(path, documents_to_save);
  } else {
    std::cerr << "File format of file " << path << " is not supported"
              << std::endl;
    return false;
  }
}

bool ProtoIO::SaveText(
    const absl::string_view path,
    const bert_annotator::Documents* documents_to_save) const {
  std::ofstream output(std::string(path), std::ios::out);
  google::protobuf::io::OstreamOutputStream fileOutput(&output,
                                                       std::ios::binary);
  if (!google::protobuf::TextFormat::Print(*documents_to_save, &fileOutput)) {
    std::cerr << "Failed to save document " << path << "." << std::endl;
    return false;
  }
  return true;
}

bool ProtoIO::SaveBinary(
    const absl::string_view path,
    const bert_annotator::Documents* documents_to_save) const {
  std::ofstream output(std::string(path),
                       std::ios::out | std::ios::trunc | std::ios::binary);
  if (!documents_to_save->SerializeToOstream(&output)) {
    std::cerr << "Failed to save document " << path << "." << std::endl;
    return false;
  }
  return true;
}

bool ProtoIO::SaveTxt(
    const absl::string_view path,
    const bert_annotator::Documents* documents_to_save) const {
  std::ofstream output(std::string(path), std::ios::out);
  if (output.is_open()) {
    for (const bert_annotator::Document& document :
         documents_to_save->documents()) {
      output << document.text() << "\n";
    }
    output.close();
    return true;
  } else {
    std::cerr << "Failed to save document " << path << "." << std::endl;
    return false;
  }
}

bert_annotator::Documents* ProtoIO::documents() {
  return &documents_;
}

}  // namespace augmenter

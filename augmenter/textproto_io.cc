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

#include "augmenter/textproto_io.h"

#include <fstream>
#include <string>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

bool TextprotoIO::load(const std::string corpus) {
  std::ifstream input("data/input/preprocessed/" + corpus + ".textproto");
  if (input.fail()) {
    std::cerr << "Failed to load corpus " << corpus << std::endl;
    return false;
  }
  google::protobuf::io::IstreamInputStream fileInput(&input, std::ios::binary);
  if (!google::protobuf::TextFormat::Parse(&fileInput, &documents_)) {
    std::cerr << "Failed to parse corpus " << corpus << std::endl;
    return false;
  }

  return true;
}

bool TextprotoIO::save(const std::string corpus) const {
  std::ofstream output("data/output/" + corpus + ".textproto");
  google::protobuf::io::OstreamOutputStream fileOutput(&output,
                                                       std::ios::binary);
  if (!google::protobuf::TextFormat::Print(documents_, &fileOutput)) {
    std::cerr << "Failed to save document." << std::endl;
    return false;
  }

  return true;
}

const bert_annotator::Documents TextprotoIO::get_documents() const {
  return documents_;
}

void TextprotoIO::set_documents(const bert_annotator::Documents documents) {
  documents_ = documents;
}

}  // namespace augmenter

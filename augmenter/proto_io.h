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

#ifndef AUGMENTER_PROTO_IO_H_
#define AUGMENTER_PROTO_IO_H_

#include <string>

#include "absl/strings/match.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class ProtoIO {
 public:
  bool Load(const absl::string_view path);
  bool Save(const absl::string_view path,
            const int output_sentences_per_file) const;
  const bert_annotator::Documents documents() const;

 private:
  bool Save(const absl::string_view path,
            const bert_annotator::Documents* documents_to_save) const;
  bool LoadTextproto(const absl::string_view path);
  bool SaveTextproto(const absl::string_view path,
                const bert_annotator::Documents* documents_to_save) const;
  bool SaveTxt(const absl::string_view path,
               const bert_annotator::Documents* documents_to_save) const;
  bool LoadBinary(const absl::string_view path);
  bool SaveBinary(const absl::string_view path,
                  const bert_annotator::Documents* documents_to_save) const;
  bert_annotator::Documents documents_;
};

}  // namespace augmenter

#endif  // AUGMENTER_PROTO_IO_H_

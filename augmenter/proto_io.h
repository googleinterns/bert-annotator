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
  bool Load(absl::string_view path);
  bool Save(absl::string_view path) const;
  const bert_annotator::Documents documents() const;
  void set_documents(const bert_annotator::Documents documents);

 private:
  bool LoadText(absl::string_view path);
  bool SaveText(absl::string_view path) const;
  bool SaveTxt(absl::string_view path) const;
  bool LoadBinary(absl::string_view path);
  bool SaveBinary(absl::string_view path) const;
  bert_annotator::Documents documents_;
};

}  // namespace augmenter

#endif  // AUGMENTER_PROTO_IO_H_

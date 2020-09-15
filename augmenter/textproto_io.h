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

#ifndef AUGMENTER_TEXTPROTO_IO_H_
#define AUGMENTER_TEXTPROTO_IO_H_

#include <string>

#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class TextprotoIO {
 public:
  bool load(std::string corpus);
  bool save(std::string corpus);
  const bert_annotator::Documents get_documents() const;
  void set_documents(bert_annotator::Documents documents);

 private:
  bert_annotator::Documents documents_;
};

}  // namespace augmenter

#endif  // AUGMENTER_TEXTPROTO_IO_H_

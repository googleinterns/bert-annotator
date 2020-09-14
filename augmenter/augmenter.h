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

#ifndef AUGMENTER_AUGMENTER_H_
#define AUGMENTER_AUGMENTER_H_
#include "augmenter/percentage.h"
#include "protocol_buffer/documents.pb.h"

class Augmenter {
 public:
  explicit Augmenter(bert_annotator::Documents documents);
  void lowercase(Percentage lowercase_percentage);
  bert_annotator::Documents get_documents();

 private:
  bert_annotator::Documents documents_;
};
#endif  // AUGMENTER_AUGMENTER_H_

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

#ifndef AUGMENTER_SHUFFLER_H_
#define AUGMENTER_SHUFFLER_H_

#include "absl/algorithm/container.h"
#include "absl/random/bit_gen_ref.h"
#include "gmock/gmock.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class Shuffler {
 public:
  virtual void Shuffle(bert_annotator::Documents* const documents,
                       absl::BitGenRef bitgenref);
};

}  // namespace augmenter

#endif  // AUGMENTER_SHUFFLER_H_

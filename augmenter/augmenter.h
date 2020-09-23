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

#include <string>
#include <utility>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "augmenter/random_sampler.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class Augmenter {
 public:
  explicit Augmenter(bert_annotator::Documents documents,
                     RandomSampler* address_sampler,
                     RandomSampler* phones_sampler);
  Augmenter(bert_annotator::Documents documents, RandomSampler* address_sampler,
            RandomSampler* phones_sampler, absl::BitGenRef bigen);
  void Augment(const int total, const int lowercase, const int addresses,
               const int phones);
  const bert_annotator::Documents documents() const;

 private:
  void Lowercase(bert_annotator::Document* const augmented_document);
  std::vector<std::pair<int, int>> DocumentBoundaryList(
      const bert_annotator::Document& document,
      const std::vector<std::string>& labels);
  void ReplaceTokens(bert_annotator::Document* document,
                     std::pair<int, int> boundaries, std::string replacement);
  bert_annotator::Documents documents_;
  RandomSampler* address_sampler_;
  RandomSampler* phones_sampler_;
  const std::vector<std::string> address_labels_;
  const std::vector<std::string> phone_labels_;
  absl::BitGenRef bitgenref_;
  absl::BitGen bitgen_;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTER_H_

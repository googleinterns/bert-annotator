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
#include "augmenter/label_boundaries.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class Augmenter {
 public:
  Augmenter(const bert_annotator::Documents& documents,
            RandomSampler* const address_sampler,
            RandomSampler* const phone_sampler);
  Augmenter(const bert_annotator::Documents& documents,
            RandomSampler* const address_sampler,
            RandomSampler* const phone_sampler, absl::BitGenRef bitgen);
  void Augment(const int augmentations_total, const int augmentations_lowercase,
               const int augmentations_address, const int augmentations_phone);
  const bert_annotator::Documents documents() const;

 private:
  bool MaybeReplaceLabel(bert_annotator::Document* const document,
                         const std::vector<std::string>& label_list,
                         const double likelihood, RandomSampler* const sampler,
                         const std::string& replacement_label);
  const std::vector<LabelBoundaries> LabelBoundaryList(
      const bert_annotator::Document& document,
      const std::vector<std::string>& labels) const;
  void ReplaceTokens(bert_annotator::Document* const document,
                     const LabelBoundaries& boundaries,
                     const std::string& replacement,
                     const std::string& replacement_label) const;
  void Lowercase(bert_annotator::Document* const document) const;
  bert_annotator::Documents documents_;
  RandomSampler* const address_sampler_;
  RandomSampler* const phone_sampler_;
  const std::vector<std::string> address_labels_;
  const std::string address_replacement_label_;
  const std::vector<std::string> phone_labels_;
  const std::string phone_replacement_label_;
  absl::BitGenRef bitgenref_;
  absl::BitGen bitgen_;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTER_H_

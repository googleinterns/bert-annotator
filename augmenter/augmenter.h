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
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "augmenter/augmentations.h"
#include "augmenter/random_sampler.h"
#include "augmenter/token_sequence.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class Augmenter {
 public:
  Augmenter(const bert_annotator::Documents& documents,
            Augmentations augmentations, RandomSampler* const address_sampler,
            RandomSampler* const phone_sampler);
  Augmenter(const bert_annotator::Documents& documents,
            Augmentations augmentations, RandomSampler* const address_sampler,
            RandomSampler* const phone_sampler, absl::BitGenRef bitgen);
  void Augment();
  const bert_annotator::Documents documents() const;

 private:
  bool AugmentAddress(bert_annotator::Document* const augmented_document);
  bool AugmentPhone(bert_annotator::Document* const augmented_document);
  bool AugmentLowercase(bert_annotator::Document* const augmented_document);
  bool MaybeReplaceLabel(const double probability, RandomSampler* const sampler,
                         const absl::string_view replacement_label,
                         bert_annotator::Document* const document);
  bool AugmentContext(bert_annotator::Document* const augmented_document);
  std::vector<TokenSequence> DropableSequences(
      const bert_annotator::Document& document);
  void DropTokens(bert_annotator::Document* const augmented_document,
                  TokenSequence boundaries) const;
  bool MaybeDropContext(const double probability,
                        bert_annotator::Document* const augmented_document);
  // Finds all token sequences labeled according to the given label list. If
  // multiple sequential tokens have different labels, but all are given in
  // the list, they are concidered to be part of the same sequence.
  const std::vector<TokenSequence> LabelBoundaryList(
      const bert_annotator::Document& document,
      const absl::string_view label) const;
  const int ReplaceText(const TokenSequence& boundaries,
                        const std::string& replacement,
                        bert_annotator::Document* const document) const;
  const int DropText(const TokenSequence& boundaries,
                     bert_annotator::Document* const document) const;
  // May introduce tokens longer than one word.
  void ReplaceToken(const int token_id, const std::string& replacement,
                    bert_annotator::Document* const document) const;
  void ShiftTokenBoundaries(const int first_token, const int shift,
                            bert_annotator::Document* const document) const;
  void ReplaceLabeledSpan(const int token_id,
                          const absl::string_view replacement_label,
                          bert_annotator::Document* const document) const;
  void UpdateLabeledSpansForDroppedTokens(
      const TokenSequence& boundaries,
      bert_annotator::Document* const document) const;
  void ReplaceLabeledTokens(const TokenSequence& boundaries,
                            const std::string& replacement,
                            const absl::string_view replacement_label,
                            bert_annotator::Document* const document) const;
  // Transforms the text to lowercase. Only explicitly listed tokens are
  // transformed.
  void Lowercase(bert_annotator::Document* const document) const;
  bert_annotator::Documents documents_;
  RandomSampler* const address_sampler_;
  RandomSampler* const phone_sampler_;
  Augmentations augmentations_;
  static const absl::flat_hash_set<absl::string_view>& kAddressLabels;
  static constexpr absl::string_view kAddressReplacementLabel = "ADDRESS";
  static const absl::flat_hash_set<absl::string_view>& kPhoneLabels;
  static constexpr absl::string_view kPhoneReplacementLabel = "TELEPHONE";
  absl::BitGenRef bitgenref_;
  absl::BitGen bitgen_;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTER_H_

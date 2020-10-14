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

#include "absl/container/flat_hash_set.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "augmenter/augmentations.h"
#include "augmenter/case_augmentation.h"
#include "augmenter/random_sampler.h"
#include "augmenter/shuffler.h"
#include "augmenter/token_range.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

class Augmenter {
 public:
  Augmenter(const bert_annotator::Documents& documents,
            Augmentations augmentations, RandomSampler* const address_sampler,
            RandomSampler* const phone_sampler, Shuffler* const shuffler,
            absl::BitGenRef bitgenref);
  void Augment();
  const bert_annotator::Documents documents() const;
  static const absl::flat_hash_set<absl::string_view>& kAddressLabels;
  static constexpr absl::string_view kAddressReplacementLabel = "ADDRESS";
  static const absl::flat_hash_set<absl::string_view>& kPhoneLabels;
  static constexpr absl::string_view kPhoneReplacementLabel = "TELEPHONE";
  // Cannot be an absl::string_view because it's used for lookups in maps that
  // expect the key to be a string.
  static const std::string& kLabelContainerName;
  static const std::vector<absl::string_view>&
      kPunctuationReplacementsWithinText;
  static const std::vector<absl::string_view>&
      kPunctuationReplacementsAtSentenceEnd;

 private:
  void AugmentAddress(bert_annotator::Document* const augmented_document);
  void AugmentPhone(bert_annotator::Document* const augmented_document);
  void AugmentCase(bert_annotator::Document* const augmented_document);
  void AugmentContextless(const absl::string_view label,
                          RandomSampler* const sampler);
  void MaybeReplaceLabel(const double probability, RandomSampler* const sampler,
                         const absl::string_view replacement_label,
                         const bool split_into_tokens,
                         bert_annotator::Document* const document);
  void AugmentPunctuation(bert_annotator::Document* const augmented_document);
  void AugmentContext(bert_annotator::Document* const augmented_document);
  // Appends the second document on the first, separated by a space.
  void ConcatenateDocument(
      bert_annotator::Document* const first_document,
      bert_annotator::Document* const second_document) const;
  // Returns the ranges of all tokens not labeled as an address or phone
  // number.
  std::vector<TokenRange> GetUnlabeledRanges(
      const bert_annotator::Document& document);
  std::vector<TokenRange> GetLabeledRanges(
      const bert_annotator::Document& document,
      absl::flat_hash_set<absl::string_view> labels);
  void DropTokens(const TokenRange boundaries,
                  bert_annotator::Document* const augmented_document) const;
  // Drops context while keeping all labels.
  void MaybeDropContextKeepLabels(
      const double probability,
      bert_annotator::Document* const augmented_document);
  // Drops context before/after a chosen label, potentially dropping other
  // labels.
  void MaybeDropContextDropLabels(
      bert_annotator::Document* const augmented_document);
  // Returns the length difference (positive = length increase).
  const int ReplaceText(const TokenRange& boundaries,
                        const std::string& replacement,
                        bert_annotator::Document* const document) const;
  // Removes the specified range of tokens from the text. Tries to keep the
  // sentence structure logical by also removing now obsolete punctuation.
  // Returns the number of deleted characters.
  const int DropText(const TokenRange& boundaries,
                     bert_annotator::Document* const document) const;
  // May introduce tokens longer than one word.
  void InsertTokens(int index, const std::vector<bert_annotator::Token> tokens,
                    bert_annotator::Document* const document) const;
  void ShiftTokenBoundaries(const int first_token, const int shift,
                            bert_annotator::Document* const document) const;
  // Drops labeled spans if associated tokens were dropped. Otherwise updates
  // the start and end indices to reflect the new token ids.
  void ShiftLabeledSpansForDroppedTokens(
      const int start, const int shift,
      bert_annotator::Document* const document) const;
  void DropLabeledSpans(const TokenRange& removed_tokens,
                        bert_annotator::Document* const document) const;
  void InsertLabeledSpan(const TokenRange& range, const absl::string_view label,
                         bert_annotator::Document* const document) const;
  int RemovePrefixPunctuation(absl::string_view* const text) const;
  int RemoveSuffixPunctuation(absl::string_view* const text) const;
  std::vector<bert_annotator::Token> SplitTextIntoTokens(
      int text_start, const absl::string_view text) const;
  void ReplaceLabeledTokens(const TokenRange& boundaries,
                            const absl::string_view replacement,
                            const absl::string_view replacement_label,
                            const bool split_into_tokens,
                            bert_annotator::Document* const document) const;
  // Changes the complete token or the first letter of a token to
  // lower/upper case. Processes only the specified tokens and returns the ids
  // of all unmodified tokens.
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>
  GetLabelListWithDefault(
      const bert_annotator::Document& document,
      google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>
          defaults_to) const;
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
  GetLabelListWithDefault(
      bert_annotator::Document* document,
      google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>*
          defaults_to) const;
  // Masks all digits with zero.
  void MaskDigits(std::string* text) const;
  void MaskDigits(bert_annotator::Document* const document) const;
  bert_annotator::Documents documents_;
  RandomSampler* const address_sampler_;
  RandomSampler* const phone_sampler_;
  Shuffler* const shuffler_;
  Augmentations augmentations_;
  absl::BitGenRef bitgenref_;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTER_H_

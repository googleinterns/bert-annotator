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

#include "augmenter/augmenter.h"

#include <map>
#include <string>
#include <vector>

#include "absl/random/mock_distributions.h"
#include "augmenter/augmentations.h"
#include "gtest/gtest.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

using ::testing::InSequence;
using ::testing::Return;
using ::testing::ReturnRef;

struct TokenSpec {
  const std::string& text;
  const int start;
  const int end;
  TokenSpec(const std::string& text_, const int start_, const int end_)
      : text(text_), start(start_), end(end_) {}
};

struct LabelSpec {
  const absl::string_view label;
  const int start;
  const int end;
  LabelSpec(const absl::string_view label_, const int start_, const int end_)
      : label(label_), start(start_), end(end_) {}
};

struct DocumentSpec {
  const absl::string_view text;
  const std::vector<TokenSpec>& token_specs;
  const bool has_labels;
  const std::map<std::string, std::vector<LabelSpec>>& label_specs_map;

  DocumentSpec(const absl::string_view text_,
               const std::vector<TokenSpec> token_specs_,
               const std::map<std::string, std::vector<LabelSpec>>
                   label_specs_map_ = {{Augmenter::kLabelContainerName, {}}})
      : text(text_),
        token_specs(token_specs_),
        has_labels(true),
        label_specs_map(label_specs_map_) {}

  DocumentSpec(
      const absl::string_view text_, const std::vector<TokenSpec> token_specs_,
      const std::map<std::string, std::vector<LabelSpec>> label_specs_map_,
      bool has_labels_)
      : text(text_),
        token_specs(token_specs_),
        has_labels(has_labels_),
        label_specs_map(label_specs_map_) {}
};

// Creates a documents wrapper with documents each containing the specified
// string and list of tokens.
bert_annotator::Documents ConstructBertDocument(
    const std::vector<DocumentSpec> document_specs) {
  bert_annotator::Documents documents = bert_annotator::Documents();
  for (const DocumentSpec document_spec : document_specs) {
    bert_annotator::Document* document = documents.add_documents();
    document->set_text(std::string(document_spec.text));
    for (const TokenSpec token_spec : document_spec.token_specs) {
      bert_annotator::Token* token;
      token = document->add_token();
      token->set_start(token_spec.start);
      token->set_end(token_spec.end);
      token->set_word(token_spec.text);
    }

    if (document_spec.has_labels) {
      bert_annotator::LabeledSpans labeled_spans = {};
      for (auto& label_pair : document_spec.label_specs_map) {
        for (auto& label_spec : label_pair.second) {
          auto labeled_span = labeled_spans.add_labeled_span();
          labeled_span->set_label(std::string(label_spec.label));
          labeled_span->set_token_start(label_spec.start);
          labeled_span->set_token_end(label_spec.end);
        }
        (*document->mutable_labeled_spans())[label_pair.first] = labeled_spans;
      }
    }
  }

  return documents;
}

augmenter::Augmentations GetDefaultAugmentations() {
  return augmenter::Augmentations{
      .num_total = 0,
      .prob_lowercasing_complete_token = 0.0,
      .prob_lowercasing_first_letter = 0.0,
      .prob_uppercasing_complete_token = 0.0,
      .prob_uppercasing_first_letter = 0.0,
      .prob_address_replacement = 0.0,
      .prob_phone_replacement = 0.0,
      .prob_context_drop_between_labels = 0.0,
      .prob_context_drop_outside_one_label = 0.0,
      .prob_punctuation_change_between_tokens = 0.0,
      .prob_punctuation_change_at_sentence_end = 0.0,
      .prob_sentence_concatenation = 0.0,
      .num_contextless_addresses = 0,
      .num_contextless_phones = 0,
      .mask_digits = false};
}

void ExpectEq(const bert_annotator::Document a,
              const bert_annotator::Document b) {
  EXPECT_STREQ(a.text().c_str(), b.text().c_str());
  ASSERT_EQ(a.token_size(), b.token_size());
  for (int i = 0; i < a.token_size(); ++i) {
    EXPECT_STREQ(a.token(i).word().c_str(), b.token(i).word().c_str());
    EXPECT_EQ(a.token(i).start(), b.token(i).start());
    EXPECT_EQ(a.token(i).end(), b.token(i).end());
  }
  ASSERT_EQ(a.labeled_spans_size(), b.labeled_spans_size());
  for (auto map_entry : a.labeled_spans()) {
    auto key = map_entry.first;
    ASSERT_EQ(a.labeled_spans().at(key).labeled_span_size(),
              b.labeled_spans().at(key).labeled_span_size());
    for (int i = 0; i < a.labeled_spans().at(key).labeled_span_size(); ++i) {
      EXPECT_STREQ(a.labeled_spans().at(key).labeled_span(i).label().c_str(),
                   b.labeled_spans().at(key).labeled_span(i).label().c_str());
      EXPECT_EQ(a.labeled_spans().at(key).labeled_span(i).token_start(),
                b.labeled_spans().at(key).labeled_span(i).token_start());
      EXPECT_EQ(a.labeled_spans().at(key).labeled_span(i).token_end(),
                b.labeled_spans().at(key).labeled_span(i).token_end());
    }
  }
}

class ShufflerStub : public Shuffler {
 public:
  void Shuffle(bert_annotator::Documents* const documents,
               absl::BitGenRef bitgenref) override {}
};

TEST(AugmenterDeathTest, NegativeProbability) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = -0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  EXPECT_DEATH(
      {
        Augmenter augmenter =
            Augmenter(documents, augmentations, &address_sampler,
                      &phone_sampler, &shuffler, bitgen);
      },
      "All probabilities must have values between zero and one.");
}

TEST(AugmenterDeathTest, ProbabilityGreaterOne) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 1.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  EXPECT_DEATH(
      {
        Augmenter augmenter =
            Augmenter(documents, augmentations, &address_sampler,
                      &phone_sampler, &shuffler, bitgen);
      },
      "All probabilities must have values between zero and one.");
}

TEST(AugmenterTest, NoAugmentation) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});

  augmenter::Augmentations augmentations = GetDefaultAugmentations();

  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  EXPECT_EQ(augmenter.documents().documents_size(), 1);
}

TEST(AugmenterDeathTest, InvalidCaseProbabilitySum) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_lowercasing_complete_token = 0.3;
  augmentations.prob_lowercasing_first_letter = 0.3;
  augmentations.prob_uppercasing_complete_token = 0.3;
  augmentations.prob_uppercasing_first_letter = 0.3;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  EXPECT_DEATH(
      {
        Augmenter augmenter =
            Augmenter(documents, augmentations, &address_sampler,
                      &phone_sampler, &shuffler, bitgen);
      },
      "The probabilities for changing the case of tokens must sum up to at "
      "most one.");
}

TEST(AugmenterTest, CreateMissingLabelList) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text", {TokenSpec("Text", 0, 3)}, {})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(0);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text", {TokenSpec("Text", 0, 3)},
                        {{Augmenter::kLabelContainerName, {}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, LowercasingCompleteTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with [Non-Token] some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 22, 25),
                     TokenSpec("InterWordCapitalization", 27, 49)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_lowercasing_complete_token = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0, 1))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.4));  // Only lowercase last token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text with [Non-Token] some interwordcapitalization",
                        {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                         TokenSpec("some", 22, 25),
                         TokenSpec("interwordcapitalization", 27, 49)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, LowercasingFirstLetter) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with [Non-Token] some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 22, 25),
                     TokenSpec("InterWordCapitalization", 27, 49)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_lowercasing_first_letter = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.4));  // Only lowercase last token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text with [Non-Token] some interWordCapitalization",
                        {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                         TokenSpec("some", 22, 25),
                         TokenSpec("interWordCapitalization", 27, 49)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, UppercasingCompleteTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with [Non-Token] some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 22, 25),
                     TokenSpec("InterWordCapitalization", 27, 49)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_uppercasing_complete_token = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.4));  // Only uppercase last token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text with [Non-Token] some INTERWORDCAPITALIZATION",
                        {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                         TokenSpec("some", 22, 25),
                         TokenSpec("INTERWORDCAPITALIZATION", 27, 49)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, UppercasingFirstLetter) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with [Non-Token] some interWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 22, 25),
                     TokenSpec("interWordCapitalization", 27, 49)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_uppercasing_first_letter = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.99))
      .WillOnce(Return(0.4));  // Only uppercase last token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text with [Non-Token] some InterWordCapitalization",
                        {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                         TokenSpec("some", 22, 25),
                         TokenSpec("InterWordCapitalization", 27, 49)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, MultipleCaseChanges) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text WITH [Non-Token] some more InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("WITH", 5, 8),
                     TokenSpec("some", 22, 25), TokenSpec("more", 27, 30),
                     TokenSpec("InterWordCapitalization", 32, 54)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_lowercasing_complete_token = 0.2;
  augmentations.prob_lowercasing_first_letter = 0.2;
  augmentations.prob_uppercasing_complete_token = 0.2;
  augmentations.prob_uppercasing_first_letter = 0.2;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(Return(0.1))   // Lowercase complete first token.
      .WillOnce(Return(0.3))   // Lowercase first letter of second token.
      .WillOnce(Return(0.5))   // Uppercase complete third token.
      .WillOnce(Return(0.7))   // Uppercase first letter of third token.
      .WillOnce(Return(0.9));  // Do not change fourth token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "text wITH [Non-Token] SOME More InterWordCapitalization",
              {TokenSpec("text", 0, 3), TokenSpec("wITH", 5, 8),
               TokenSpec("SOME", 22, 25), TokenSpec("More", 27, 30),
               TokenSpec("InterWordCapitalization", 32, 54)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneLongerLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98765432109876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 98765432109876543210! Thanks.",
              {TokenSpec("Call", 0, 3),
               TokenSpec("98765432109876543210", 5, 24),
               TokenSpec("Thanks", 27, 32)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneShorterLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 98! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("98", 5, 6),
               TokenSpec("Thanks", 9, 14)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneStart) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "0123456789! Thanks.",
      {TokenSpec("0123456789", 0, 9), TokenSpec("Thanks", 12, 17)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "9876543210! Thanks.",
              {TokenSpec("9876543210", 0, 9), TokenSpec("Thanks", 12, 17)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneEnd) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneChooseLabel) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("0123456789 or 0123456789",
                    {TokenSpec("0123456789", 0, 9), TokenSpec("or", 11, 12),
                     TokenSpec("0123456789", 14, 23)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 2;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample())
      .Times(2)
      .WillRepeatedly(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .Times(2)
      .WillRepeatedly(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_phone_replacement))
      .WillOnce(Return(false))
      .WillOnce(Return(true))
      .WillOnce(Return(true))
      .WillOnce(Return(false));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "9876543210 or 0123456789",
              {TokenSpec("9876543210", 0, 9), TokenSpec("or", 11, 12),
               TokenSpec("0123456789", 14, 23)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                 LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
  augmented = augmenter.documents().documents(2);
  expected = ConstructBertDocument(
                 {DocumentSpec(
                     "0123456789 or 9876543210",
                     {TokenSpec("0123456789", 0, 9), TokenSpec("or", 11, 12),
                      TokenSpec("9876543210", 14, 23)},
                     {{Augmenter::kLabelContainerName,
                       {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                        LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})})
                 .documents(0);
  ExpectEq(augmented, expected);
}

// Replacing addresses is very similiar to replacing phone numbers, so not all
// respective tests are repeated here.
TEST(AugmenterTest, UpdateLabels) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("Thanks", 13, 18)},
      {{Augmenter::kLabelContainerName, {LabelSpec("LOCALITY", 1, 1)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_phone_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const auto augmented = augmenter.documents().documents(0);
  const auto expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Zurich! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
               TokenSpec("Thanks", 13, 18)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("Thanks", 14, 19)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("OTHER", 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Munich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
                         TokenSpec("Thanks", 14, 19)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1),
                           LabelSpec("OTHER", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressFewerTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich City! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("City", 13, 16), TokenSpec("Thanks", 19, 24)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2),
                       LabelSpec("OTHER", 3, 3)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Munich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
                         TokenSpec("Thanks", 14, 19)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1),
                           LabelSpec("OTHER", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceMultipleAddressesFewerTokens) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich City! Thanks. Other city",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("City", 13, 16), TokenSpec("Thanks", 19, 24),
       TokenSpec("Other", 27, 31), TokenSpec("city", 33, 36)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2),
         LabelSpec("OTHER", 3, 3), LabelSpec("LOCALITY", 4, 5)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample())
      .Times(2)
      .WillRepeatedly(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .Times(2)
      .WillRepeatedly(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich! Thanks. Munich",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
               TokenSpec("Thanks", 14, 19), TokenSpec("Munich", 22, 27)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1),
                 LabelSpec("OTHER", 2, 2),
                 LabelSpec(Augmenter::kAddressReplacementLabel, 3, 3)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressMultiWordReplacement) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich City! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("City", 13, 16), TokenSpec("Thanks", 19, 24)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2),
                       LabelSpec("OTHER", 3, 3)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich Centrum";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich Centrum! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
               TokenSpec("Centrum", 13, 19), TokenSpec("Thanks", 22, 27)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 2),
                 LabelSpec("OTHER", 3, 3)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressMoreTokensReplacement) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("Thanks", 14, 19)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("OTHER", 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich Centrum";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich Centrum! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
               TokenSpec("Centrum", 13, 19), TokenSpec("Thanks", 22, 27)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 2),
                 LabelSpec("OTHER", 3, 3)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressPunctuation) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("Thanks", 14, 19)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("OTHER", 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Str. A, 1 - a";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Str. A, 1 - a! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Str", 6, 8),
                         TokenSpec("A", 11, 11), TokenSpec("1", 14, 14),
                         TokenSpec("a", 18, 18), TokenSpec("Thanks", 21, 26)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 4),
                           LabelSpec("OTHER", 5, 5)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDetectMultipleDroppableSequences) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
                     TokenSpec("0123456789", 14, 23),
                     TokenSpec("Postfix", 25, 31), TokenSpec("tokens", 33, 38)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .Times(2)
      .WillRepeatedly(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();
}

TEST(AugmenterTest, DropContextStartAndEnd) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Prefix tokens 0123456789 Middle tokens 0123456789 Postfix tokens.",
      {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
       TokenSpec("0123456789", 14, 23), TokenSpec("Middle", 25, 30),
       TokenSpec("tokens", 32, 37), TokenSpec("0123456789", 39, 48),
       TokenSpec("Postfix", 50, 56), TokenSpec("tokens", 58, 63)},
      {{Augmenter::kLabelContainerName,
        {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2),
         LabelSpec(Augmenter::kPhoneReplacementLabel, 5, 5)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .Times(2)
      .WillRepeatedly(
          Return(0));  // 1 x use first document, 1 x drop first token.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .Times(3)
      .WillRepeatedly(
          Return(true));  // Drop from all three sequences (second sequence
                          // will be skipped as it is too short).
  // Drop "tokens" (last).
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 7, 7))
      .WillOnce(Return(7));
  // Dropping "Prefix" is defined above.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "tokens 0123456789 Middle tokens 0123456789 Postfix.",
              {TokenSpec("tokens", 0, 5), TokenSpec("0123456789", 7, 16),
               TokenSpec("Middle", 18, 23), TokenSpec("tokens", 25, 30),
               TokenSpec("0123456789", 32, 41), TokenSpec("Postfix", 43, 49)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1),
                 LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextRemoveBeginningOfLabel) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
                     TokenSpec("0123456789", 14, 23),
                     TokenSpec("Postfix", 25, 31), TokenSpec("tokens", 33, 38)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec("OTHER", 0, 1),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .Times(2)  // 1 x use first document, 1 x drop first token.
      .WillRepeatedly(Return(0));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .WillOnce(Return(false))  // Do not drop from second sequence.
      .WillOnce(Return(true));  //  Drop from first sequence.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "tokens 0123456789 Postfix tokens.",
              {TokenSpec("tokens", 0, 5), TokenSpec("0123456789", 7, 16),
               TokenSpec("Postfix", 18, 24), TokenSpec("tokens", 26, 31)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextRemoveMiddleOfLabel) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("0 Many prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
                     TokenSpec("prefix", 7, 12), TokenSpec("tokens", 14, 19),
                     TokenSpec("0123456789", 21, 30),
                     TokenSpec("Postfix", 32, 38), TokenSpec("tokens", 40, 45)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .WillOnce(Return(false))  // Do not drop from second sequence.
      .WillOnce(Return(true));  //  Drop from first sequence.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 2, 2))
      .Times(2)
      .WillRepeatedly(Return(2));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "0 Many tokens 0123456789 Postfix tokens.",
              {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
               TokenSpec("tokens", 7, 12), TokenSpec("0123456789", 14, 23),
               TokenSpec("Postfix", 25, 31), TokenSpec("tokens", 33, 38)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                 LabelSpec(Augmenter::kPhoneReplacementLabel, 3, 3)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextRemoveEndOfLabel) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
                     TokenSpec("0123456789", 14, 23),
                     TokenSpec("Postfix", 25, 31), TokenSpec("tokens", 33, 38)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2),
                       LabelSpec("OTHER", 3, 4)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .WillOnce(Return(true))    // Drop from second sequence.
      .WillOnce(Return(false));  //  Do not drop from first sequence.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 4, 4))
      .WillOnce(Return(4));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Prefix tokens 0123456789 Postfix.",
              {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
               TokenSpec("0123456789", 14, 23), TokenSpec("Postfix", 25, 31)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextNoLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_between_labels = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_between_labels))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDropLabelsNoLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_outside_one_label = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen,
                   augmentations.prob_context_drop_outside_one_label))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDropLabelsPrefix) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("0 Many prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
                     TokenSpec("prefix", 7, 12), TokenSpec("tokens", 14, 19),
                     TokenSpec("0123456789", 21, 30),
                     TokenSpec("Postfix", 32, 38), TokenSpec("tokens", 40, 45)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_outside_one_label = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 2))
      .WillOnce(Return(1));  // Keep second label.
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_outside_one_label))
      .Times(2)  // Drop beginning and end.
      .WillRepeatedly(Return(true));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 6, 6))
      .WillOnce(Return(6));  // Drop last token.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));  // Drop first token.
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Many prefix tokens 0123456789 Postfix.",
              {TokenSpec("Many", 0, 3), TokenSpec("prefix", 5, 10),
               TokenSpec("tokens", 12, 17), TokenSpec("0123456789", 19, 28),
               TokenSpec("Postfix", 30, 36)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec("OTHER", 0, 2),
                 LabelSpec(Augmenter::kPhoneReplacementLabel, 3, 3)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDropLabelsSuffix) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("0 Many prefix tokens 0123456789 Postfix tokens.",
                    {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
                     TokenSpec("prefix", 7, 12), TokenSpec("tokens", 14, 19),
                     TokenSpec("0123456789", 21, 30),
                     TokenSpec("Postfix", 32, 38), TokenSpec("tokens", 40, 45)},
                    {{Augmenter::kLabelContainerName,
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_context_drop_outside_one_label = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 2))
      .WillOnce(Return(0));  // Keep first label.
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_context_drop_outside_one_label))
      .WillOnce(Return(true));  //  Drop end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 2, 6))
      .WillOnce(Return(3));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "0 Many prefix.",
              {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
               TokenSpec("prefix", 7, 12)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ChangePunctuationBetweenWords) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text - with, some! more punctuation.",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 7, 10),
                     TokenSpec("some", 13, 16), TokenSpec("more", 19, 22),
                     TokenSpec("punctuation", 24, 34)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_punctuation_change_between_tokens = 1;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(
      absl::MockBernoulli(),
      Call(bitgen, augmentations.prob_punctuation_change_between_tokens))
      .WillRepeatedly(Return(true));
  EXPECT_CALL(
      absl::MockUniform<int>(),
      Call(bitgen, 0, Augmenter::kPunctuationReplacementsWithinText.size()))
      .WillOnce(Return(0))
      .WillOnce(Return(1))
      .WillOnce(Return(2))
      .WillOnce(Return(3));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text, with; some: more - punctuation.",
                        {TokenSpec("Text", 0, 3), TokenSpec("with", 6, 9),
                         TokenSpec("some", 12, 15), TokenSpec("more", 18, 21),
                         TokenSpec("punctuation", 25, 35)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ChangePunctuationAtSentenceEnd) {
  bert_annotator::Documents documents =
      ConstructBertDocument({DocumentSpec("Text", {TokenSpec("Text", 0, 3)}),
                             DocumentSpec("Text!", {TokenSpec("Text", 0, 3)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 6;
  augmentations.prob_punctuation_change_at_sentence_end = 1;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 1))
      .WillOnce(Return(0))
      .WillOnce(Return(0))
      .WillRepeatedly(Return(1));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(
      absl::MockBernoulli(),
      Call(bitgen, augmentations.prob_punctuation_change_at_sentence_end))
      .WillRepeatedly(Return(true));
  EXPECT_CALL(
      absl::MockUniform<int>(),
      Call(bitgen, 0, Augmenter::kPunctuationReplacementsAtSentenceEnd.size()))
      .WillOnce(Return(0))
      .WillOnce(Return(1))
      .WillOnce(Return(2))
      .WillOnce(Return(3))
      .WillOnce(Return(4))
      .WillOnce(Return(5));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(2);
  bert_annotator::Document expected =
      ConstructBertDocument({DocumentSpec("Text?", {TokenSpec("Text", 0, 3)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(3);
  expected =
      ConstructBertDocument({DocumentSpec("Text!", {TokenSpec("Text", 0, 3)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(4);
  expected =
      ConstructBertDocument({DocumentSpec("Text.", {TokenSpec("Text", 0, 3)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(5);
  expected =
      ConstructBertDocument({DocumentSpec("Text:", {TokenSpec("Text", 0, 3)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(6);
  expected =
      ConstructBertDocument({DocumentSpec("Text;", {TokenSpec("Text", 0, 3)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(7);
  expected = ConstructBertDocument(
                 {DocumentSpec("Text - ", {TokenSpec("Text", 0, 3)})})
                 .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ChangePunctuationAtSentenceEndNoTokens) {
  bert_annotator::Documents documents =
      ConstructBertDocument({DocumentSpec("...", {})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_punctuation_change_at_sentence_end = 1;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(
      absl::MockBernoulli(),
      Call(bitgen, augmentations.prob_punctuation_change_at_sentence_end))
      .WillRepeatedly(Return(true));
  EXPECT_CALL(
      absl::MockUniform<int>(),
      Call(bitgen, 0, Augmenter::kPunctuationReplacementsAtSentenceEnd.size()))
      .Times(0);
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument({DocumentSpec("...", {})}).documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, RemoveSeparatorTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text, more ... t.e.x.t.!",
                    {TokenSpec("Text", 0, 3), TokenSpec(",", 4, 4),
                     TokenSpec("more", 6, 9), TokenSpec("...", 11, 13),
                     TokenSpec("t.e.x.t.", 15, 22), TokenSpec("!", 23, 23)})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(0);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Text, more ... t.e.x.t.!",
                        {TokenSpec("Text", 0, 3), TokenSpec("more", 6, 9),
                         TokenSpec("t.e.x.t.", 15, 22)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, MergeDocuments) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec(
           "Some labeled text.",
           {TokenSpec("Some", 0, 3), TokenSpec("labeled", 5, 11),
            TokenSpec("text", 13, 16)},
           {{Augmenter::kLabelContainerName, {LabelSpec("OTHER_A", 1, 1)}}}),
       DocumentSpec(
           "Some more labeled text.",
           {TokenSpec("Some", 0, 3), TokenSpec("more", 5, 8),
            TokenSpec("labeled", 10, 16), TokenSpec("text", 18, 21)},
           {{Augmenter::kLabelContainerName, {LabelSpec("OTHER_B", 1, 2)}}})});

  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.prob_sentence_concatenation = 0.5;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_sentence_concatenation))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);
  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 3);
  bert_annotator::Document augmented = augmenter.documents().documents(2);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Some labeled text. Some more labeled text.",
              {TokenSpec("Some", 0, 3), TokenSpec("labeled", 5, 11),
               TokenSpec("text", 13, 16), TokenSpec("Some", 19, 22),
               TokenSpec("more", 24, 27), TokenSpec("labeled", 29, 35),
               TokenSpec("text", 37, 40)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec("OTHER_A", 1, 1), LabelSpec("OTHER_B", 4, 5)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, MaskDigits) {
  // This document contains numbers within tokens, tokens solely consisting of
  // numbers and numbers outside of tokens. The label will be replaced with
  // additional numbers.
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Text with [LABEL] num_0123_bers 99 99",
      {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
       TokenSpec("[LABEL]", 10, 16), TokenSpec("num_0123_bers", 18, 30),
       TokenSpec("99", 32, 33)},
      {{Augmenter::kLabelContainerName, {LabelSpec("LOCALITY", 2, 2)}}})});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.prob_address_replacement = 0.5;
  augmentations.mask_digits = true;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Addr.01";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(),
              Call(bitgen, augmentations.prob_address_replacement))
      .WillOnce(Return(true));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(0);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Text with [LABEL] num_0000_bers 00 00",
              {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
               TokenSpec("[LABEL]", 10, 16), TokenSpec("num_0000_bers", 18, 30),
               TokenSpec("00", 32, 33)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec("ADDRESS", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(1);
  expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Text with Addr.00 num_0000_bers 00 00",
              {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
               TokenSpec("Addr.00", 10, 16), TokenSpec("num_0000_bers", 18, 30),
               TokenSpec("00", 32, 33)},
              {{Augmenter::kLabelContainerName,
                {LabelSpec("ADDRESS", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ContextlessAddress) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.num_contextless_addresses = 1;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Sample Address 1";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(0);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Sample Address 1",
                        {TokenSpec("Sample", 0, 5), TokenSpec("Address", 7, 13),
                         TokenSpec("1", 15, 15)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec("AD"
                                     "DR"
                                     "ES"
                                     "S",
                                     0, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ContextlessAddressChangeCaseAndPunctuation) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.num_contextless_addresses = 1;
  augmentations.prob_lowercasing_complete_token = 0.9;
  augmentations.prob_punctuation_change_between_tokens = 0.8;
  augmentations.prob_punctuation_change_at_sentence_end = 0.7;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Sample Address 1";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0, 1))
      .Times(3)
      .WillRepeatedly(Return(0.5));  // Lowercase all tokens.
  EXPECT_CALL(
      absl::MockBernoulli(),
      Call(bitgen, augmentations.prob_punctuation_change_between_tokens))
      .Times(2)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(
      absl::MockUniform<int>(),
      Call(bitgen, 0, Augmenter::kPunctuationReplacementsWithinText.size()))
      .Times(2)
      .WillRepeatedly(Return(0));
  EXPECT_CALL(
      absl::MockBernoulli(),
      Call(bitgen, augmentations.prob_punctuation_change_at_sentence_end))
      .WillOnce(Return(true));
  EXPECT_CALL(
      absl::MockUniform<int>(),
      Call(bitgen, 0, Augmenter::kPunctuationReplacementsAtSentenceEnd.size()))
      .WillOnce(Return(0));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(0);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("sample, address, 1?",
                        {TokenSpec("sample", 0, 5), TokenSpec("address", 8, 14),
                         TokenSpec("1", 17, 17)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec("ADDRESS", 0, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ContextlessPhone) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.num_contextless_phones = 1;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "0123456789";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(0);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("0123456789", {TokenSpec("0123456789", 0, 9)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec("TELEPHONE", 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ContextlessPhoneMaskedDigits) {
  bert_annotator::Documents documents = ConstructBertDocument({});
  augmenter::Augmentations augmentations = GetDefaultAugmentations();
  augmentations.num_total = 1;
  augmentations.num_contextless_phones = 1;
  augmentations.mask_digits = true;
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "0123456789";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  ShufflerStub shuffler;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, &shuffler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(0);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("0000000000", {TokenSpec("0000000000", 0, 9)},
                        {{Augmenter::kLabelContainerName,
                          {LabelSpec("TELEPHONE", 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

}  // namespace augmenter

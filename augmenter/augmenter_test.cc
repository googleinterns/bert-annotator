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
                   label_specs_map_ = {{"lucid", {}}})
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
      std::map<std::string, bert_annotator::LabeledSpans> label_map = {};
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

TEST(AugmenterTest, NoAugmentation) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {.num_total = 0,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_EQ(augmenter.documents().documents_size(), 1);
}

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_EQ(augmenter.documents().documents_size(), 2);
}

TEST(AugmenterTest, NoAugmentations) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {.num_total = 10,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  for (int i = 0; i < augmentations.num_total + 1; ++i) {
    ExpectEq(augmenter.documents().documents(i), documents.documents(0));
  }
}

TEST(AugmenterTest, Lowercasing) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});
  Augmentations augmentations = {.num_total = 10,
                                 .num_lowercasings = 10,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_STREQ(augmenter.documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization");
  EXPECT_STREQ(augmenter.documents().documents(0).token(0).word().c_str(),
               "Text");
  EXPECT_STREQ(augmenter.documents().documents(0).token(1).word().c_str(),
               "with");
  EXPECT_STREQ(augmenter.documents().documents(0).token(2).word().c_str(),
               "some");
  EXPECT_STREQ(augmenter.documents().documents(0).token(3).word().c_str(),
               "InterWordCapitalization");
  for (int i = 1; i < augmentations.num_total + 1; ++i) {
    EXPECT_STREQ(augmenter.documents().documents(i).text().c_str(),
                 "text with some interwordcapitalization");
    EXPECT_STREQ(augmenter.documents().documents(i).token(0).word().c_str(),
                 "text");
    EXPECT_STREQ(augmenter.documents().documents(i).token(1).word().c_str(),
                 "with");
    EXPECT_STREQ(augmenter.documents().documents(i).token(2).word().c_str(),
                 "some");
    EXPECT_STREQ(augmenter.documents().documents(i).token(3).word().c_str(),
                 "interwordcapitalization");
  }
}

TEST(AugmenterTest, RandomizedLowercasing) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization [0]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)}),
       DocumentSpec("Text with some InterWordCapitalization [1]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)}),
       DocumentSpec("Text with some InterWordCapitalization [2]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});

  absl::MockingBitGen bitgen;
  // Document sampling.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0))
      .WillOnce(Return(1))
      .WillOnce(Return(2))
      .WillOnce(Return(0));
  // Phone/Address replacement probability.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  // Lowercasing.
  InSequence s;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 2.0 / 4))
      .WillOnce(Return(true));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 3))
      .WillOnce(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 2))
      .WillOnce(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 1))
      .WillOnce(Return(true));
  Augmentations augmentations = {.num_total = 4,
                                 .num_lowercasings = 2,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(3);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("text with some interwordcapitalization [0]",
                        {TokenSpec("text", 0, 3), TokenSpec("with", 5, 8),
                         TokenSpec("some", 10, 13),
                         TokenSpec("interwordcapitalization", 15, 37)})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(6);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(4);
  expected = documents.documents(1);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(5);
  expected = documents.documents(2);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DontLowercaseNonTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("[BOS] Text with some InterWordCapitalization [EOS]",
                    {TokenSpec("Text", 6, 9), TokenSpec("with", 11, 14),
                     TokenSpec("some", 16, 19),
                     TokenSpec("InterWordCapitalization", 21, 43)})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 1,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("[BOS] text with some interwordcapitalization [EOS]",
                        {TokenSpec("text", 6, 9), TokenSpec("with", 11, 14),
                         TokenSpec("some", 16, 19),
                         TokenSpec("interwordcapitalization", 21, 43)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DontReplacePhone) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789! Thanks.",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
       TokenSpec("Thanks", 17, 22)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected = documents.documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789! Thanks.",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
       TokenSpec("Thanks", 17, 22)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x address, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneLongerLength) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789! Thanks.",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
       TokenSpec("Thanks", 17, 22)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98765432109876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x address, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 98765432109876543210! Thanks.",
              {TokenSpec("Call", 0, 3),
               TokenSpec("98765432109876543210", 5, 24),
               TokenSpec("Thanks", 27, 32)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneShorterLength) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789! Thanks.",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
       TokenSpec("Thanks", 17, 22)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x address, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 98! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("98", 5, 6),
               TokenSpec("Thanks", 9, 14)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneStart) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "0123456789! Thanks.",
      {TokenSpec("0123456789", 0, 9), TokenSpec("Thanks", 12, 17)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x address, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "9876543210! Thanks.",
              {TokenSpec("9876543210", 0, 9), TokenSpec("Thanks", 12, 17)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneEnd) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x address, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneChooseLabel) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("0123456789 or 0123456789",
                    {TokenSpec("0123456789", 0, 9), TokenSpec("or", 11, 12),
                     TokenSpec("0123456789", 14, 23)},
                    {{"lucid",
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  Augmentations augmentations = {.num_total = 2,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 2,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample())
      .Times(2)
      .WillRepeatedly(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(8)  // 2 x address, 2 x lowercasing, 4 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .Times(2)  // 2 x phone.
      .WillRepeatedly(Return(true));
  // Token selection.
  EXPECT_CALL(absl::MockUniform<size_t>(),
              Call(absl::IntervalClosed, bitgen, 0, 1))
      .WillOnce(Return(0))
      .WillOnce(Return(1));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "9876543210 or 0123456789",
              {TokenSpec("9876543210", 0, 9), TokenSpec("or", 11, 12),
               TokenSpec("0123456789", 14, 23)},
              {{"lucid",
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
                     {{"lucid",
                       {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                        LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})})
                 .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneChooseDocument) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Call 0123456789! Thanks.",
      {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
       TokenSpec("Thanks", 17, 22)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 2,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(9)  // 2 x address, 2 x lowercasing, 1 x phone, 4 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(true));  // Replace phone in first document.
  // Token selection.
  EXPECT_CALL(absl::MockUniform<size_t>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
  augmented = augmenter.documents().documents(2);
  expected = documents.documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneMissingLucid) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("MissingLucid", {TokenSpec("MissingLucid", 0, 12)}, {}),
       DocumentSpec(
           "Call 0123456789! Thanks.",
           {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
            TokenSpec("Thanks", 17, 22)},
           {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 1,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  // Document sampling.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 1))
      .WillOnce(Return(0))
      .WillOnce(Return(1));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(8)  // 2 x address, 2 x lowercasing, 4 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .Times(2)  // 2 x phone.
      .WillRepeatedly(Return(true));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 3);
  const bert_annotator::Document augmented = augmenter.documents().documents(2);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

// Replacing addresses is very similiar to replacing phone numbers, so not all
// respective tests are repeated here.
TEST(AugmenterTest, DontReplaceAddress) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 2);
  const auto augmented = augmenter.documents().documents(1);
  const auto expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Zurich! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
               TokenSpec("Thanks", 13, 18)},
              {{"lucid",
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, UpdateLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {.num_total = 0,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  const auto augmented = augmenter.documents().documents(0);
  const auto expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Zurich! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
               TokenSpec("Thanks", 13, 18)},
              {{"lucid",
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 1,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x address.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
               TokenSpec("Thanks", 13, 18)},
              {{"lucid",
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressFewerTokens) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich City! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("City", 13, 16), TokenSpec("Thanks", 18, 23)},
      {{"lucid",
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 1,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x address.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
               TokenSpec("Thanks", 13, 18)},
              {{"lucid",
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressMultiWordReplacement) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich City! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("City", 13, 16), TokenSpec("Thanks", 18, 23)},
      {{"lucid",
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 1,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich Centrum";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x address.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich Centrum! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich Centrum", 6, 19),
               TokenSpec("Thanks", 21, 26)},
              {{"lucid",
                {LabelSpec(Augmenter::kAddressReplacementLabel, 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDetectMultipleDroppableSequences) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Prefix tokens 0123456789 Postfix tokens.",
      {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
       TokenSpec("0123456789", 14, 23), TokenSpec("Postfix", 25, 31),
       TokenSpec("tokens", 33, 38)},
      {{"lucid", {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  // Drop from both sequences.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();
}

TEST(AugmenterTest, DropContextStartAndEnd) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Prefix tokens 0123456789 Middle tokens 0123456789 Postfix tokens.",
      {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
       TokenSpec("0123456789", 14, 23), TokenSpec("Middle", 25, 30),
       TokenSpec("tokens", 32, 37), TokenSpec("0123456789", 39, 48),
       TokenSpec("Postfix", 50, 56), TokenSpec("tokens", 58, 63)},
      {{"lucid",
        {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2),
         LabelSpec(Augmenter::kPhoneReplacementLabel, 5, 5)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .Times(2)
      .WillRepeatedly(
          Return(0));  // 1 x use first document, 1 x drop first token.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(3)
      .WillRepeatedly(
          Return(true));  // Drop from all three sequences (second sequence will
                          // be skipped as it is too short).
  // Drop "tokens" (last).
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 7, 7))
      .WillOnce(Return(7));
  // Dropping "Prefix" is defined above.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "tokens 0123456789 Middle tokens 0123456789 Postfix.",
              {TokenSpec("tokens", 0, 5), TokenSpec("0123456789", 7, 16),
               TokenSpec("Middle", 18, 23), TokenSpec("tokens", 25, 30),
               TokenSpec("0123456789", 32, 41), TokenSpec("Postfix", 43, 49)},
              {{"lucid",
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
                    {{"lucid",
                      {LabelSpec("OTHER", 0, 1),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .Times(2)  // 1 x use first document, 1 x drop first token.
      .WillRepeatedly(Return(0));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(false))  // Do not drop from second sequence.
      .WillOnce(Return(true));  //  Drop from first sequence.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "tokens 0123456789 Postfix tokens.",
              {TokenSpec("tokens", 0, 5), TokenSpec("0123456789", 7, 16),
               TokenSpec("Postfix", 18, 24), TokenSpec("tokens", 26, 31)},
              {{"lucid",
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
                    {{"lucid",
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(false))  // Do not drop from second sequence.
      .WillOnce(Return(true));  //  Drop from first sequence.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 2, 2))
      .Times(2)
      .WillRepeatedly(Return(2));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "0 Many tokens 0123456789 Postfix tokens.",
              {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
               TokenSpec("tokens", 7, 12), TokenSpec("0123456789", 14, 23),
               TokenSpec("Postfix", 25, 31), TokenSpec("tokens", 33, 38)},
              {{"lucid",
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
                    {{"lucid",
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2),
                       LabelSpec("OTHER", 3, 4)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(true))    // Drop from second sequence.
      .WillOnce(Return(false));  //  Do not drop from first sequence.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 4, 4))
      .WillOnce(Return(4));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Prefix tokens 0123456789 Postfix.",
              {TokenSpec("Prefix", 0, 5), TokenSpec("tokens", 7, 12),
               TokenSpec("0123456789", 14, 23), TokenSpec("Postfix", 25, 31)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextNoLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextNoLabelsNoLucid) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)},
                    {})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 1,
                                 .num_context_drops_outside_one_label = 0,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(3)  // 1 x phone, 1 x addresss, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)},
                        {})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDropLabelsNoLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 1,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x addresss, 1 x lowercasing, 1 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, DropContextDropLabelsNoLabelsNoLucid) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text without any tokens.",
                    {TokenSpec("Text", 0, 3), TokenSpec("without", 5, 11),
                     TokenSpec("any", 13, 15), TokenSpec("tokens", 17, 22)},
                    {})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 1,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x addresss, 1 x lowercasing, 1 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 1, 3))
      .WillOnce(Return(3));
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("without any.",
                        {TokenSpec("without", 0, 6), TokenSpec("any", 8, 10)},
                        {})})
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
                    {{"lucid",
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 1,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x addresss, 1 x lowercasing, 1 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 2))
      .WillOnce(Return(1));  // Keep second label.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .Times(2)
      .WillRepeatedly(Return(true));  //  Drop from beginning and end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 6, 6))
      .WillOnce(Return(6));  // Drop last token.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 2))
      .WillOnce(Return(0));  // Drop first token.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Many prefix tokens 0123456789 Postfix.",
              {TokenSpec("Many", 0, 3), TokenSpec("prefix", 5, 10),
               TokenSpec("tokens", 12, 17), TokenSpec("0123456789", 19, 28),
               TokenSpec("Postfix", 30, 36)},
              {{"lucid",
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
                    {{"lucid",
                      {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0),
                       LabelSpec("OTHER", 1, 3),
                       LabelSpec(Augmenter::kPhoneReplacementLabel, 4, 4)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 1,
                                 .probability_per_drop = 0.5,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 0, 0))
      .WillOnce(Return(0));  // Use first document.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x addresss, 1 x lowercasing, 1 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x context.
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 2))
      .WillOnce(Return(0));  // Keep first label.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(true));  //  Drop end.
  EXPECT_CALL(absl::MockUniform<int>(),
              Call(absl::IntervalClosed, bitgen, 2, 6))
      .WillOnce(Return(3));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "0 Many prefix.",
              {TokenSpec("0", 0, 0), TokenSpec("Many", 2, 5),
               TokenSpec("prefix", 7, 12)},
              {{"lucid",
                {LabelSpec(Augmenter::kPhoneReplacementLabel, 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, RemoveSeparatorTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text, more ... t.e.x.t.!",
                    {TokenSpec("Text", 0, 3), TokenSpec(",", 4, 4),
                     TokenSpec("more", 6, 9), TokenSpec("...", 11, 13),
                     TokenSpec("t.e.x.t.", 15, 22), TokenSpec("!", 23, 23)})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 0,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .mask_digits = false};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  absl::MockingBitGen bitgen;

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

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

TEST(AugmenterTest, MaskDigits) {
  // This document contains numbers within tokens, tokens solely consisting of
  // numbers and numbers outside of tokens. The label will be replaced with
  // additional numbers.
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Text with [LABEL] num_0123_bers 99 99",
      {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
       TokenSpec("[LABEL]", 10, 16), TokenSpec("num_0123_bers", 18, 30),
       TokenSpec("99", 32, 33)},
      {{"lucid", {LabelSpec("LOCALITY", 2, 2)}}})});
  Augmentations augmentations = {.num_total = 1,
                                 .num_lowercasings = 0,
                                 .num_address_replacements = 1,
                                 .num_phone_replacements = 0,
                                 .num_context_drops_between_labels = 0,
                                 .num_context_drops_outside_one_label = 0,
                                 .mask_digits = true};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Addr. 1";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 1 x phone, 1 x lowercasing, 2 x context.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x address.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(0);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Text with [LABEL] num_0000_bers 00 00",
              {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
               TokenSpec("[LABEL]", 10, 16), TokenSpec("num_0000_bers", 18, 30),
               TokenSpec("00", 32, 33)},
              {{"lucid", {LabelSpec("ADDRESS", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);

  augmented = augmenter.documents().documents(1);
  expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Text with Addr. 0 num_0000_bers 00 00",
              {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
               TokenSpec("Addr. 0", 10, 16), TokenSpec("num_0000_bers", 18, 30),
               TokenSpec("00", 32, 33)},
              {{"lucid", {LabelSpec("ADDRESS", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

}  // namespace augmenter

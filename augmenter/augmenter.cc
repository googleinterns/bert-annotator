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

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "augmenter/augmentations.h"
#include "augmenter/case_augmentation.h"
#include "augmenter/random_sampler.h"
#include "augmenter/shuffler.h"
#include "augmenter/token_range.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

Augmenter::Augmenter(bert_annotator::Documents* documents,
                     Augmentations augmentations,
                     RandomSampler* const address_sampler,
                     RandomSampler* const phone_sampler,
                     Shuffler* const shuffler, absl::BitGenRef bitgenref)
    : documents_(documents),
      address_sampler_(address_sampler),
      phone_sampler_(phone_sampler),
      shuffler_(shuffler),
      augmentations_(augmentations),
      bitgenref_(bitgenref) {
  // Assert valid probabilities.
  const std::vector<double> probabilities = {
      augmentations_.prob_lowercasing_complete_token,
      augmentations_.prob_lowercasing_first_letter,
      augmentations_.prob_uppercasing_complete_token,
      augmentations_.prob_uppercasing_first_letter,
      augmentations_.prob_address_replacement,
      augmentations_.prob_phone_replacement,
      augmentations_.prob_context_drop_between_labels,
      augmentations_.prob_context_drop_outside_one_label};
  if (absl::c_any_of(probabilities, [](double probability) {
        return probability < 0.0 || probability > 1.0;
      })) {
    std::cerr << "All probabilities must have values between zero and one."
              << std::endl;
    abort();
  }
  if (augmentations_.prob_lowercasing_complete_token +
          augmentations_.prob_lowercasing_first_letter +
          augmentations_.prob_uppercasing_complete_token +
          augmentations_.prob_uppercasing_first_letter >
      1) {
    std::cerr << "The probabilities for changing the case of tokens must sum "
                 "up to at most one."
              << std::endl;
    abort();
  }

  // Skip invalid sentences where the start/end value of tokens does not
  // match their length.
  for (int i = documents_->documents_size() - 1; i >= 0; --i) {
    bert_annotator::Document* document = documents_->mutable_documents(i);

    for (bert_annotator::Token token : document->token()) {
      if (static_cast<int>(token.word().size()) !=
          token.end() - token.start() + 1) {
        documents_->mutable_documents()->erase(
            documents_->mutable_documents()->begin() + i);
        break;
      }
    }
  }

  for (bert_annotator::Document& document : *documents_->mutable_documents()) {
    InitializeLabelList(&document);
    MergePhoneNumberTokens(&document);
    DropSeparatorTokens(&document);
    UnifyAndMergeAddressLabels(&document);
  }
}

void Augmenter::AugmentAddress(
    bert_annotator::Document* const augmented_document) {
  MaybeReplaceLabel(augmentations_.prob_address_replacement, address_sampler_,
                    Augmenter::kAddressReplacementLabel,
                    /*split_into_tokens=*/true, augmented_document);
}

void Augmenter::AugmentPhone(
    bert_annotator::Document* const augmented_document) {
  MaybeReplaceLabel(augmentations_.prob_phone_replacement, phone_sampler_,
                    Augmenter::kPhoneReplacementLabel,
                    /*split_into_tokens=*/false, augmented_document);
}

void Augmenter::AugmentCase(bert_annotator::Document* const document) {
  std::string* const text = document->mutable_text();
  std::string new_text;
  int text_index = 0;
  for (bert_annotator::Token& token : *document->mutable_token()) {
    std::string* const word = token.mutable_word();

    // Adds the non-tokens before the current token as they are.
    const int token_start = token.start();
    const int token_end = token.end();
    if (text_index < token_start) {
      new_text.append(text->begin() + text_index, text->begin() + token_start);
    }

    // Sample one of the five actions (lower/upper case complete token/first
    // letter, keep unmodified). Using a discrete distribution would be cleaner,
    // but cannot be mocked in tests.
    double random_value = absl::Uniform<double>(bitgenref_, 0.0, 1.0);
    double boundary_lowercase_complete_token =
        augmentations_.prob_lowercasing_complete_token;
    double boundary_lowercase_first_letter =
        boundary_lowercase_complete_token +
        augmentations_.prob_lowercasing_first_letter;
    double boundary_uppercase_complete_token =
        boundary_lowercase_first_letter +
        augmentations_.prob_uppercasing_complete_token;
    double boundary_uppercase_first_letter =
        boundary_uppercase_complete_token +
        augmentations_.prob_uppercasing_first_letter;

    if (random_value < boundary_lowercase_complete_token) {
      absl::AsciiStrToLower(word);
    } else if (random_value < boundary_lowercase_first_letter) {
      word->at(0) = std::tolower(word->at(0));
    } else if (random_value < boundary_uppercase_complete_token) {
      absl::AsciiStrToUpper(word);
    } else if (random_value < boundary_uppercase_first_letter) {
      word->at(0) = std::toupper(word->at(0));
    }

    new_text.append(*word);
    text_index = token_end + 1;
  }
  new_text.append(text->begin() + text_index, text->end());
  document->set_text(new_text);
}

bert_annotator::Document* Augmenter::AugmentContextless(
    const absl::string_view label, const bool split_into_tokens,
    RandomSampler* const sampler) {
  bert_annotator::Document* document = documents_->add_documents();

  const std::string sample = sampler->Sample();
  document->set_text(sample);
  std::vector<bert_annotator::Token> new_tokens;
  if (split_into_tokens) {
    new_tokens = SplitTextIntoTokens(0, sample);
  } else {
    bert_annotator::Token token;
    token.set_word(sample);
    token.set_start(0);
    token.set_end(sample.size() - 1);
    new_tokens.push_back(token);
  }
  InsertTokens(0, new_tokens, document);
  InitializeLabelList(document);
  InsertLabeledSpan(
      TokenRange{.start = 0, .end = static_cast<int>(new_tokens.size()) - 1},
      label, document);

  return document;
}

void Augmenter::Augment() {
  const int original_document_number = documents_->documents_size();

  for (int i = 0; i < augmentations_.num_total; ++i) {
    if (i % 100000 == 0) {
      std::cout << "Augmentation progress: "
                << std::to_string(i * 100 / augmentations_.num_total) << "%"
                << std::endl;
    }
    bert_annotator::Document* augmented_document;

    if (i < augmentations_.num_contextless_addresses) {
      augmented_document =
          AugmentContextless(kAddressReplacementLabel,
                             /*split_into_tokens=*/true, address_sampler_);
    } else if (i < augmentations_.num_contextless_addresses +
                       augmentations_.num_contextless_phones) {
      augmented_document = AugmentContextless(
          kPhoneReplacementLabel, /*split_into_tokens=*/false, phone_sampler_);
    } else {
      const int document_id = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                            original_document_number - 1);
      const bert_annotator::Document& original_document =
          documents_->documents(document_id);
      augmented_document = documents_->add_documents();
      augmented_document->CopyFrom(original_document);

      AugmentAddress(augmented_document);
      AugmentPhone(augmented_document);
      AugmentContext(augmented_document);
    }

    AugmentCase(augmented_document);
    AugmentPunctuation(augmented_document);
  }

  // If the documents are not shuffled, the unmodified sentences are all next to
  // each other. This is problematic when merging sentences in the next step.
  if (augmentations_.shuffle) {
    std::cout << "Shuffling..." << std::endl;
    shuffler_->Shuffle(documents_, bitgenref_);
  }

  std::cout << "Concatenating..." << std::endl;
  for (int i = documents_->documents_size() - 1; i > 0; --i) {
    if (absl::Bernoulli(bitgenref_,
                        augmentations_.prob_sentence_concatenation)) {
      AddConcatenatedDocument(documents_->documents(i - 1),
                              documents_->documents(i));
    }
  }

  // Shuffle again, now that concatenated sentences have been added.
  if (augmentations_.shuffle) {
    std::cout << "Shuffling..." << std::endl;
    shuffler_->Shuffle(documents_, bitgenref_);
  }

  if (augmentations_.mask_digits) {
    std::cout << "Maskinkg digits..." << std::endl;
    for (bert_annotator::Document& document :
         *documents_->mutable_documents()) {
      MaskDigits(&document);
    }
  }

  for (const bert_annotator::Document& document : documents_->documents()) {
    if (document.text().length() == 0) {
      std::cerr << "Empty document in output detected. This will break the "
                   "evaluation scripts, aborting."
                << std::endl;
      abort();
    }
  }
}

void Augmenter::AddConcatenatedDocument(
    const bert_annotator::Document& first_document,
    const bert_annotator::Document& second_document) {
  bert_annotator::Document* concatenated_document = documents_->add_documents();
  concatenated_document->CopyFrom(first_document);
  bert_annotator::Document tmp_copy_of_second_document;
  tmp_copy_of_second_document.CopyFrom(second_document);

  ShiftTokenBoundaries(0, first_document.text().size() + 1,
                       &tmp_copy_of_second_document);
  ShiftLabeledSpansForDroppedTokens(0, first_document.token_size(),
                                    &tmp_copy_of_second_document);
  concatenated_document->mutable_text()->append(
      " " + tmp_copy_of_second_document.text());
  InsertTokens(concatenated_document->token_size(),
               std::vector<bert_annotator::Token>(
                   tmp_copy_of_second_document.token().begin(),
                   tmp_copy_of_second_document.token().end()),
               concatenated_document);
  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelList(tmp_copy_of_second_document);
  for (const bert_annotator::LabeledSpan& labeled_span : labeled_spans) {
    InsertLabeledSpan(TokenRange{.start = labeled_span.token_start(),
                                 .end = labeled_span.token_end()},
                      labeled_span.label(), concatenated_document);
  }
}

void Augmenter::AugmentContext(
    bert_annotator::Document* const augmented_document) {
  MaybeDropContextKeepLabels(augmentations_.prob_context_drop_between_labels,
                             augmented_document);
  MaybeDropContextDropLabels(augmented_document);
}

void Augmenter::AugmentPunctuation(bert_annotator::Document* const document) {
  for (int i = 0; i < document->token_size() - 1; ++i) {
    bert_annotator::Token* const token = document->mutable_token(i);
    const bool do_change_punctuation = absl::Bernoulli(
        bitgenref_, augmentations_.prob_punctuation_change_between_tokens);
    if (!do_change_punctuation) {
      continue;
    }
    const int punctuation_replacement_id = absl::Uniform(
        bitgenref_, 0,
        static_cast<int>(kPunctuationReplacementsWithinText.size()));
    const absl::string_view punctuation_replacement =
        kPunctuationReplacementsWithinText[punctuation_replacement_id];
    const int current_punctuation_length =
        document->token(i + 1).start() - token->end() - 1;
    document->mutable_text()->replace(token->end() + 1,
                                      current_punctuation_length,
                                      std::string(punctuation_replacement));
    ShiftTokenBoundaries(
        i + 1, punctuation_replacement.size() - current_punctuation_length,
        document);
  }

  const bool do_change_punctuation = absl::Bernoulli(
      bitgenref_, augmentations_.prob_punctuation_change_at_sentence_end);
  if (do_change_punctuation && document->token_size() > 0) {
    const int punctuation_replacement_id = absl::Uniform(
        bitgenref_, 0,
        static_cast<int>(kPunctuationReplacementsAtSentenceEnd.size()));
    const absl::string_view punctuation_replacement =
        kPunctuationReplacementsAtSentenceEnd[punctuation_replacement_id];
    bert_annotator::Token last_token =
        document->token(document->token_size() - 1);
    const int current_punctuation_length =
        document->text().size() - last_token.end() - 1;
    document->mutable_text()->replace(last_token.end() + 1,
                                      current_punctuation_length,
                                      std::string(punctuation_replacement));
  }
}

std::vector<TokenRange> Augmenter::GetUnlabeledRanges(
    const bert_annotator::Document& document) {
  std::vector<TokenRange> droppable_ranges;
  int start = 0;

  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelList(document);
  for (const bert_annotator::LabeledSpan& labeled_span : labeled_spans) {
    if (labeled_span.label() == Augmenter::kAddressReplacementLabel ||
        labeled_span.label() == Augmenter::kPhoneReplacementLabel) {
      if (start != labeled_span.token_start()) {
        droppable_ranges.push_back(
            TokenRange{.start = start, .end = labeled_span.token_start() - 1});
      }
      start = labeled_span.token_end() + 1;
    }
  }
  if (start != document.token_size()) {
    droppable_ranges.push_back(
        TokenRange{.start = start, .end = document.token_size() - 1});
  }

  return droppable_ranges;
}

std::vector<TokenRange> Augmenter::GetLabeledRanges(
    const bert_annotator::Document& document,
    absl::flat_hash_set<absl::string_view> labels) {
  std::vector<TokenRange> labeled_ranges;
  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelList(document);
  for (const bert_annotator::LabeledSpan& labeled_span : labeled_spans) {
    if (labels.contains(labeled_span.label())) {
      labeled_ranges.push_back(TokenRange{.start = labeled_span.token_start(),
                                          .end = labeled_span.token_end()});
    }
  }
  return labeled_ranges;
}

void Augmenter::MaybeDropContextKeepLabels(
    const double probability,
    bert_annotator::Document* const augmented_document) {
  std::vector<TokenRange> droppable_ranges =
      GetUnlabeledRanges(*augmented_document);
  // If there are no labels, we will get a single droppable range containing
  // the whole sentence. To drop from both its start and end, we duplicate it
  // here.
  bool no_labels = false;
  if (droppable_ranges.size() == 1 && droppable_ranges[0].start == 0 &&
      droppable_ranges[0].end == augmented_document->token_size() - 1) {
    no_labels = true;
    droppable_ranges.push_back(droppable_ranges[0]);
  }

  // Tokens will be dropped, so iterating backwards avoids the need to
  // update the indices in droppable_ranges.
  for (int i = droppable_ranges.size() - 1; i >= 0; --i) {
    const TokenRange& droppable_range = droppable_ranges[i];
    const bool do_drop = absl::Bernoulli(bitgenref_, probability);
    if (!do_drop) {
      continue;
    }

    // At least one token should be dropped, so the range needs to be at
    // least two tokens long.
    if (droppable_range.end == droppable_range.start) {
      continue;
    }

    int drop_tokens_start;
    int drop_tokens_end;
    // For context after the last label, drop a postfix of the sentence.
    if (i == static_cast<int>(droppable_ranges.size()) - 1 &&
        droppable_range.end == augmented_document->token_size() - 1) {
      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        droppable_range.start + 1, droppable_range.end);
      drop_tokens_end = droppable_range.end;
      // If no labels exist, the range was duplicated, so the index needs to
      // be updated.
      if (no_labels) {
        droppable_ranges[0].end = drop_tokens_start - 1;
      }
    }
    // For context before the first label, drop a prefix of the sentence. //
    else if (droppable_range.start == 0) {  // NOLINT
      drop_tokens_start = 0;
      drop_tokens_end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                      droppable_range.end - 1);
    }
    // For context between two labels, drop a subsequence.
    else {  // NOLINT
      // At least the first and the last tokens must not be dropped.
      if (droppable_range.end - droppable_range.start == 1) {
        continue;
      }

      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        droppable_range.start + 1, droppable_range.end - 1);
      drop_tokens_end =
          absl::Uniform(absl::IntervalClosed, bitgenref_, drop_tokens_start,
                        droppable_range.end - 1);
    }

    const TokenRange boundaries =
        TokenRange{.start = drop_tokens_start, .end = drop_tokens_end};
    const int removed_characters = DropText(boundaries, augmented_document);
    DropTokens(boundaries, augmented_document);
    ShiftTokenBoundaries(boundaries.start, -removed_characters,
                         augmented_document);
    DropLabeledSpans(boundaries, augmented_document);
    ShiftLabeledSpansForDroppedTokens(
        drop_tokens_end + 1, -(drop_tokens_end - drop_tokens_start + 1),
        augmented_document);
  }
}

void Augmenter::MaybeDropContextDropLabels(
    bert_annotator::Document* const augmented_document) {
  if (augmentations_.prob_context_drop_outside_one_label == 0) {
    return;
  }

  const int token_count = augmented_document->token_size();
  std::vector<TokenRange> labeled_ranges = GetLabeledRanges(
      *augmented_document, {kAddressReplacementLabel, kPhoneReplacementLabel});
  // MaybeDropContextKeepLabels already implements dropping from sentences
  // without any labels.
  if (labeled_ranges.size() == 0) {
    return MaybeDropContextKeepLabels(
        augmentations_.prob_context_drop_outside_one_label, augmented_document);
  }

  const int label_id =
      absl::Uniform(bitgenref_, 0, static_cast<int>(labeled_ranges.size()));
  TokenRange labeled_range = labeled_ranges[label_id];
  if (labeled_range.end < token_count - 2) {
    TokenRange drop_range{.start = 0, .end = token_count - 1};
    const bool drop_right = absl::Bernoulli(
        bitgenref_, augmentations_.prob_context_drop_outside_one_label);
    if (drop_right) {
      drop_range.start = absl::Uniform(absl::IntervalClosed, bitgenref_,
                                       labeled_range.end + 2, token_count - 1);
      const int removed_characters = DropText(drop_range, augmented_document);
      DropTokens(drop_range, augmented_document);
      ShiftTokenBoundaries(drop_range.start, -removed_characters,
                           augmented_document);
      DropLabeledSpans(drop_range, augmented_document);
      ShiftLabeledSpansForDroppedTokens(
          drop_range.end + 1, -(drop_range.end - drop_range.start + 1),
          augmented_document);
    }
  }
  if (labeled_range.start > 1) {
    TokenRange drop_range{.start = 0, .end = 0};
    const bool drop_left = absl::Bernoulli(
        bitgenref_, augmentations_.prob_context_drop_outside_one_label);
    if (drop_left) {
      drop_range.end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                     labeled_range.start - 2);
      const int removed_characters = DropText(drop_range, augmented_document);
      DropTokens(drop_range, augmented_document);
      ShiftTokenBoundaries(drop_range.start, -removed_characters,
                           augmented_document);
      DropLabeledSpans(drop_range, augmented_document);
      ShiftLabeledSpansForDroppedTokens(
          drop_range.end + 1, -(drop_range.end - drop_range.start + 1),
          augmented_document);
    }
  }
}

void Augmenter::MaybeReplaceLabel(const double probability,
                                  RandomSampler* const sampler,
                                  const absl::string_view label,
                                  const bool split_into_tokens,
                                  bert_annotator::Document* const document) {
  const std::vector<TokenRange>& labeled_ranges =
      GetLabeledRanges(*document, {label});

  // The replacement invalidates the boundaries of labeled ranges that occur to
  // the right, so it has to be done right to left.
  for (int i = labeled_ranges.size() - 1; i >= 0; --i) {
    TokenRange labeled_range = labeled_ranges[i];
    if (!absl::Bernoulli(bitgenref_, probability)) {
      continue;
    }
    const std::string replacement = sampler->Sample();
    ReplaceLabeledTokens(labeled_range, replacement, label, split_into_tokens,
                         document);
  }
}

void Augmenter::MergePhoneNumberTokens(
    bert_annotator::Document* const document) const {
  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelList(*document);
  for (int label_index = labeled_spans.size() - 1; label_index >= 0;
       --label_index) {
    const bert_annotator::LabeledSpan& labeled_span =
        labeled_spans.at(label_index);
    const int token_start = labeled_span.token_start();
    const int token_end = labeled_span.token_end();
    if (labeled_span.label() == kPhoneReplacementLabel) {
      std::string merged_token_text;
      for (int token_index = token_start; token_index <= token_end;
           ++token_index) {
        if (token_index > token_start) {
          for (int intermediate_char_index =
                   document->token(token_index - 1).end() + 1;
               intermediate_char_index < document->token(token_index).start();
               ++intermediate_char_index) {
            absl::StrAppend(
                &merged_token_text,
                document->text().substr(intermediate_char_index, 1));
          }
        }
        absl::StrAppend(&merged_token_text,
                        document->token(token_index).word());
      }
      bert_annotator::Token merged_token;
      merged_token.set_start(document->token(token_start).start());
      merged_token.set_end(document->token(token_end).end());
      merged_token.set_word(merged_token_text);

      DropTokens(TokenRange{.start = token_start, .end = token_end}, document);
      InsertTokens(token_start, {merged_token}, document);
      ShiftLabeledSpansForDroppedTokens(token_start + 1,
                                        -(token_end - token_start), document);
    }
  }
}

void Augmenter::DropSeparatorTokens(
    bert_annotator::Document* const document) const {
  for (int i = document->token_size() - 1; i >= 0; --i) {
    const bert_annotator::Token& token = document->token(i);
    // TODO(brix): depends on the installed C locale, may need to be changed
    // for non-english languages.
    if (absl::c_none_of(token.word(), [](unsigned char c) {
          return std::isdigit(c) || std::isalpha(c);
        })) {
      const TokenRange removed_token = TokenRange{.start = i, .end = i};
      DropTokens(removed_token, document);

      // If this was the only token of a label (i.e., the whole label consisted
      // of non-alphanumeric values, such as emoticons), the label should be
      // dropped. If the label remains, its bounds have to be updated manually,
      // as ShiftLabeledSpansForDroppedTokens would update both start and end.
      int label_shift_start = removed_token.start;
      google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
          labeled_spans = GetLabelList(document);
      for (bert_annotator::LabeledSpan& labeled_span : *labeled_spans) {
        if (labeled_span.token_start() == i && labeled_span.token_end() == i) {
          DropLabeledSpans(removed_token, document);
        } else if (labeled_span.token_start() <= i &&
                   labeled_span.token_end() >= i) {
          labeled_span.set_token_end(labeled_span.token_end() - 1);
          label_shift_start = labeled_span.token_end() + 1;
        }
      }

      ShiftLabeledSpansForDroppedTokens(label_shift_start, -1, document);
    }
  }
}

void Augmenter::UnifyAndMergeAddressLabels(
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelList(document);
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    bert_annotator::LabeledSpan& labeled_span = labeled_spans->at(i);
    if (kAddressLabels.contains(labeled_span.label())) {
      labeled_span.set_label(std::string(Augmenter::kAddressReplacementLabel));

      // Consecutive address labels should be merged.
      if (i <= labeled_spans->size() - 2) {
        bert_annotator::LabeledSpan& successor_span = labeled_spans->at(i + 1);
        if (successor_span.label() == Augmenter::kAddressReplacementLabel &&
            labeled_span.token_end() + 1 == successor_span.token_start()) {
          labeled_span.set_token_end(successor_span.token_end());
          labeled_spans->erase(labeled_spans->begin() + i + 1);
        }
      }
    }
  }
}

const int Augmenter::ReplaceText(
    const TokenRange& boundaries, const std::string& replacement,
    bert_annotator::Document* const document) const {
  const int string_start = document->token(boundaries.start).start();
  const int string_end = document->token(boundaries.end).end();
  std::string new_text;
  new_text.append(document->mutable_text()->begin(),
                  document->mutable_text()->begin() + string_start);
  new_text.append(replacement);
  if (static_cast<int>(document->text().size()) > string_end) {
    new_text.append(document->mutable_text()->begin() + string_end + 1,
                    document->mutable_text()->end());
  }
  document->set_text(new_text);

  return replacement.size() - (string_end - string_start + 1);
}

const int Augmenter::DropText(const TokenRange& boundaries,
                              bert_annotator::Document* const document) const {
  int text_start;
  int text_end;
  if (boundaries.start > 0) {
    text_start = document->token(boundaries.start - 1).end() + 1;
    text_end = document->token(boundaries.end).end();
  } else {
    text_start = 0;
    if (boundaries.end < document->token_size() - 1) {
      text_end = document->token(boundaries.end + 1).start() - 1;
    } else {
      text_end = document->text().size() - 1;
    }
  }

  std::string new_text;
  new_text.append(document->mutable_text()->begin(),
                  document->mutable_text()->begin() + text_start);
  if (static_cast<int>(document->text().size()) > text_end) {
    new_text.append(document->mutable_text()->begin() + text_end + 1,
                    document->mutable_text()->end());
  }
  document->set_text(new_text);

  return text_end - text_start + 1;
}

void Augmenter::InsertTokens(int index,
                             const std::vector<bert_annotator::Token> tokens,
                             bert_annotator::Document* const document) const {
  for (bert_annotator::Token token : tokens) {
    bert_annotator::Token* const new_token = document->add_token();
    new_token->CopyFrom(token);

    for (int i = document->token_size() - 1; i > index; --i) {
      document->mutable_token()->SwapElements(i, i - 1);
    }
    ++index;
  }
}

void Augmenter::DropTokens(const TokenRange boundaries,
                           bert_annotator::Document* const document) const {
  document->mutable_token()->erase(
      document->mutable_token()->begin() + boundaries.start,
      document->mutable_token()->begin() + boundaries.end +
          1);  // end is inclusive.
}

void Augmenter::ShiftTokenBoundaries(
    const int first_token, const int shift,
    bert_annotator::Document* const document) const {
  for (int i = first_token; i < document->token_size(); ++i) {
    bert_annotator::Token* token = document->mutable_token(i);
    token->set_start(token->start() + shift);
    token->set_end(token->end() + shift);
  }
}

void Augmenter::DropLabeledSpans(
    const TokenRange& removed_tokens,
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelList(document);
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    bert_annotator::LabeledSpan* const labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_end() >= removed_tokens.start &&
        labeled_span->token_start() <= removed_tokens.end) {
      labeled_spans->erase(labeled_spans->begin() + i);
    }
  }
}

void Augmenter::InsertLabeledSpan(
    const TokenRange& range, const absl::string_view label,
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelList(document);
  bert_annotator::LabeledSpan* const new_span = labeled_spans->Add();
  new_span->set_label(std::string(label));
  new_span->set_token_start(range.start);
  new_span->set_token_end(range.end);

  int i = labeled_spans->size() - 1;
  while (i > 0 && labeled_spans->at(i).token_start() <
                      labeled_spans->at(i - 1).token_end()) {
    labeled_spans->SwapElements(i, i - 1);
    --i;
  }
}

void Augmenter::ShiftLabeledSpansForDroppedTokens(
    const int start, const int shift,
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelList(document);
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    bert_annotator::LabeledSpan* const labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_start() >= start) {
      labeled_span->set_token_start(labeled_span->token_start() + shift);
    }
    if (labeled_span->token_end() >= start) {
      labeled_span->set_token_end(labeled_span->token_end() + shift);
    }
  }
}

int Augmenter::RemovePrefixPunctuation(absl::string_view* const text) const {
  int removed_punctuation = 0;
  bool punctuation_removed = true;
  while (punctuation_removed && text->size() > 0) {
    punctuation_removed = false;
    const char first_char = text->at(0);
    if (!std::isalpha(first_char) && !std::isdigit(first_char)) {
      text->remove_prefix(1);
      ++removed_punctuation;
      punctuation_removed = true;
    }
  }
  return removed_punctuation;
}

int Augmenter::RemoveSuffixPunctuation(absl::string_view* const text) const {
  int removed_punctuation = 0;
  bool punctuation_removed = true;
  while (punctuation_removed && text->size() > 0) {
    punctuation_removed = false;
    const char last_char = text->at(text->size() - 1);
    if (!std::isalpha(last_char) && !std::isdigit(last_char)) {
      text->remove_suffix(1);
      ++removed_punctuation;
      punctuation_removed = true;
    }
  }
  return removed_punctuation;
}

std::vector<bert_annotator::Token> Augmenter::SplitTextIntoTokens(
    int text_start, const absl::string_view text) const {
  std::vector<bert_annotator::Token> new_tokens;
  int replacement_text_index = 0;
  for (absl::string_view potential_token_text : absl::StrSplit(text, " ")) {
    replacement_text_index += RemovePrefixPunctuation(&potential_token_text);
    int trailing_punctuation = RemoveSuffixPunctuation(&potential_token_text);

    if (potential_token_text.size() > 0) {
      bert_annotator::Token token;
      token.set_word(std::string(potential_token_text));
      token.set_start(text_start + replacement_text_index);
      token.set_end(text_start + replacement_text_index +
                    potential_token_text.size() - 1);
      new_tokens.push_back(token);
    }
    replacement_text_index +=
        potential_token_text.size() + 1 + trailing_punctuation;
  }

  return new_tokens;
}

void Augmenter::ReplaceLabeledTokens(
    const TokenRange& boundaries, const absl::string_view replacement,
    const absl::string_view replacement_label, const bool split_into_tokens,
    bert_annotator::Document* const document) const {
  const int text_start = document->token(boundaries.start).start();
  std::vector<bert_annotator::Token> new_tokens;
  if (split_into_tokens) {
    new_tokens = SplitTextIntoTokens(text_start, replacement);
  } else {
    bert_annotator::Token token;
    token.set_word(std::string(replacement));
    token.set_start(text_start);
    token.set_end(text_start + replacement.size() - 1);
    new_tokens.push_back(token);
  }

  const int text_shift =
      ReplaceText(boundaries, std::string(replacement), document);
  DropTokens(boundaries, document);
  InsertTokens(boundaries.start, new_tokens, document);
  ShiftTokenBoundaries(boundaries.start + new_tokens.size(), text_shift,
                       document);
  DropLabeledSpans(boundaries, document);
  const int token_number_increase =
      new_tokens.size() - (boundaries.end - boundaries.start + 1);
  // The remaining tokens should be shifted first, so they do not overlap with
  // the new ones.
  ShiftLabeledSpansForDroppedTokens(boundaries.end + 1, token_number_increase,
                                    document);
  InsertLabeledSpan(TokenRange{.start = boundaries.start,
                               .end = boundaries.start +
                                      static_cast<int>(new_tokens.size()) - 1},
                    replacement_label, document);
}

void Augmenter::InitializeLabelList(
    bert_annotator::Document* const document) const {
  if (document->labeled_spans().find(kLabelContainerName) ==
      document->labeled_spans().end()) {
    (*document->mutable_labeled_spans())[kLabelContainerName] = {};
  }
}

google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>
Augmenter::GetLabelList(const bert_annotator::Document& document) const {
  return document.labeled_spans().at(kLabelContainerName).labeled_span();
}

google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
Augmenter::GetLabelList(bert_annotator::Document* document) const {
  return document->mutable_labeled_spans()
      ->at(kLabelContainerName)
      .mutable_labeled_span();
}

void Augmenter::MaskDigits(std::string* text) const {
  absl::StrReplaceAll({{"1", "0"},
                       {"2", "0"},
                       {"3", "0"},
                       {"4", "0"},
                       {"5", "0"},
                       {"6", "0"},
                       {"7", "0"},
                       {"8", "0"},
                       {"9", "0"}},
                      text);
}

void Augmenter::MaskDigits(bert_annotator::Document* const document) const {
  MaskDigits(document->mutable_text());

  for (bert_annotator::Token& token : *document->mutable_token()) {
    MaskDigits(token.mutable_word());
  }
}

const absl::flat_hash_set<absl::string_view>& Augmenter::kAddressLabels =
    *new absl::flat_hash_set<absl::string_view>(
        {"LOCALITY", "COUNTRY", "ADMINISTRATIVE_AREA", "THOROUGHFARE",
         "THOROUGHFARE_NUMBER", "PREMISE", "POSTAL_CODE", "PREMISE_LEVEL"});

constexpr absl::string_view Augmenter::kAddressReplacementLabel;

const absl::flat_hash_set<absl::string_view>& Augmenter::kPhoneLabels = {
    "TELEPHONE"};

constexpr absl::string_view Augmenter::kPhoneReplacementLabel;

const std::string& Augmenter::kLabelContainerName = *new std::string("lucid");

const std::vector<absl::string_view>&
    Augmenter::kPunctuationReplacementsWithinText =
        *new std::vector<absl::string_view>({", ", "; ", ": ", " - "});

const std::vector<absl::string_view>&
    Augmenter::kPunctuationReplacementsAtSentenceEnd =
        *new std::vector<absl::string_view>({"?", "!", ".", ":", ";", " - "});

}  // namespace augmenter

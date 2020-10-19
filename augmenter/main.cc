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

#include <fstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "augmenter/augmentations.h"
#include "augmenter/augmenter.h"
#include "augmenter/proto_io.h"
#include "augmenter/random_sampler.h"
#include "augmenter/shuffler.h"

ABSL_FLAG(std::vector<std::string>, corpora, std::vector<std::string>({}),
          "comma-separated list of corpora to augment");
ABSL_FLAG(std::string, addresses_path, "",
          "Path to list of alternative addresses");
ABSL_FLAG(std::string, phones_path, "",
          "Path to list of alternative phone number");
ABSL_FLAG(int, num_total, 0, "Number of total augmentations");

ABSL_FLAG(double, prob_lowercasing_complete_token, 0,
          "Probability of lowercasing a complete token");
ABSL_FLAG(double, prob_lowercasing_first_letter, 0,
          "Probability of lowercasing the first letter of a token");
ABSL_FLAG(double, prob_uppercasing_complete_token, 0,
          "Probability of uppercasing a complete token");
ABSL_FLAG(double, prob_uppercasing_first_letter, 0,
          "Probability of uppercasing the first letter of a token");
ABSL_FLAG(double, prob_address_replacement, 0,
          "Probability of replacing an address");
ABSL_FLAG(double, prob_phone_replacement, 0,
          "Probability of replacing a phone number");
ABSL_FLAG(double, prob_context_drop_between_labels, 0,
          "Probability of dropping context in between labels. Keeps at least "
          "the token directly to the left and right of each label");
ABSL_FLAG(double, prob_context_drop_outside_one_label, 0,
          "Probability of selecting a label and dropping context to its left "
          "and right. May drop other labels");
ABSL_FLAG(double, prob_punctuation_change_between_tokens, 0,
          "Probability of changing the punctuation between tokens to be one of "
          "{\", \", \": \", \"; \", \" - \"}");
ABSL_FLAG(double, prob_punctuation_change_at_sentence_end, 0,
          "Probability of changing the punctuation at the sentence end to be "
          "one of {\"?\", \"!\", \".\", \":\", \";\", \" - \"}");
ABSL_FLAG(double, prob_sentence_concatenation, 0,
          "Probability of concatenating sentences");
ABSL_FLAG(
    int, num_contextless_addresses, 0,
    "Number of sentences solely consisting of an address, without any context");
ABSL_FLAG(int, num_contextless_phones, 0,
          "Number of sentences solely consisting of a phone number, without "
          "any context");
ABSL_FLAG(bool, mask_digits, false,
          "If set, all digits are replaced with zeros");
ABSL_FLAG(bool, save_as_text, false,
          "If set, the augmented data is saved in text format. Useful for "
          "debugging");

// Augments the dataset by applying configurable actions, see defined flags.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  absl::ParseCommandLine(argc, argv);

  const std::vector<std::string> corpora = absl::GetFlag(FLAGS_corpora);
  const std::string addresses_path = absl::GetFlag(FLAGS_addresses_path);
  const std::string phones_path = absl::GetFlag(FLAGS_phones_path);

  augmenter::Augmentations augmentations{
      .num_total = absl::GetFlag(FLAGS_num_total),
      .prob_lowercasing_complete_token =
          absl::GetFlag(FLAGS_prob_lowercasing_complete_token),
      .prob_lowercasing_first_letter =
          absl::GetFlag(FLAGS_prob_lowercasing_first_letter),
      .prob_uppercasing_complete_token =
          absl::GetFlag(FLAGS_prob_uppercasing_complete_token),
      .prob_uppercasing_first_letter =
          absl::GetFlag(FLAGS_prob_uppercasing_first_letter),
      .prob_address_replacement = absl::GetFlag(FLAGS_prob_address_replacement),
      .prob_phone_replacement = absl::GetFlag(FLAGS_prob_phone_replacement),
      .prob_context_drop_between_labels =
          absl::GetFlag(FLAGS_prob_context_drop_between_labels),
      .prob_context_drop_outside_one_label =
          absl::GetFlag(FLAGS_prob_context_drop_outside_one_label),
      .prob_punctuation_change_between_tokens =
          absl::GetFlag(FLAGS_prob_punctuation_change_between_tokens),
      .prob_punctuation_change_at_sentence_end =
          absl::GetFlag(FLAGS_prob_punctuation_change_at_sentence_end),
      .prob_sentence_concatenation =
          absl::GetFlag(FLAGS_prob_sentence_concatenation),
      .num_contextless_addresses =
          absl::GetFlag(FLAGS_num_contextless_addresses),
      .num_contextless_phones = absl::GetFlag(FLAGS_num_contextless_phones),
      .mask_digits = absl::GetFlag(FLAGS_mask_digits)};

  for (const std::string& corpus : corpora) {
    augmenter::ProtoIO textproto_io = augmenter::ProtoIO();
    if (!textproto_io.LoadText(corpus)) {
      std::cerr << "Skipping corpus " << corpus << "." << std::endl;
      continue;
    }
    std::cout << "Processing corpus " << corpus << std::endl;

    if (addresses_path.empty()) {
      std::cerr << "List of alternative addresses must be provided."
                << std::endl;
      return false;
    }
    std::ifstream address_stream(addresses_path);
    if (address_stream.fail()) {
      std::cerr << "Failed to load address file " << addresses_path
                << std::endl;
      return false;
    }
    augmenter::RandomSampler address_sampler(address_stream);

    if (phones_path.empty()) {
      std::cerr << "List of alternative phone numbers must be provided."
                << std::endl;
      return false;
    }

    std::ifstream phones_stream(phones_path);
    if (phones_stream.fail()) {
      std::cerr << "Failed to load phone number file " << phones_path
                << std::endl;
      return false;
    }
    augmenter::RandomSampler phones_sampler(phones_stream);

    augmenter::Shuffler shuffler;

    absl::BitGen bitgen;

    augmenter::Augmenter augmenter = augmenter::Augmenter(
        textproto_io.documents(), augmentations, &address_sampler,
        &phones_sampler, &shuffler, bitgen);
    augmenter.Augment();
    textproto_io.set_documents(augmenter.documents());
    if (absl::GetFlag(FLAGS_save_as_text)) {
      textproto_io.SaveText(corpus);
    } else {
      textproto_io.SaveBinary(corpus);
    }
  }

  return 0;
}

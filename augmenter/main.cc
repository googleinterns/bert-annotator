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
#include "augmenter/random_sampler.h"
#include "augmenter/textproto_io.h"

ABSL_FLAG(int, num_total, 0, "Number of created augmented samples");
ABSL_FLAG(std::vector<std::string>, corpora, std::vector<std::string>({}),
          "comma-separated list of corpora to augment");
ABSL_FLAG(int, num_lowercasings, 0, "Number of augmentations by lowercasing");
ABSL_FLAG(std::string, addresses_path, "",
          "Path to list of alternative addresses");
ABSL_FLAG(int, num_address_replacements, 0,
          "Number of augmentations by address replacement");
ABSL_FLAG(std::string, phones_path, "",
          "Path to list of alternative phone number");
ABSL_FLAG(int, num_phone_replacements, 0,
          "Number of augmentations by phone number replacement");
ABSL_FLAG(
    int, num_context_drops_between_labels, 0,
    "Number of augmentations by dropping context in between labels. Keeps at "
    "least the token directly to the left and right of each label.");
ABSL_FLAG(int, num_context_drops_outside_one_label, 0,
          "Number of augmentations by selecting a label and dropping context "
          "to its left and right. May drop other labels.");
ABSL_FLAG(double, probability_per_drop, 0.5,
          "Given that context from a sentence will be dropped, how likely is "
          "each sequence to be dropped?");
ABSL_FLAG(bool, mask_digits, false,
          "If true, all digits are replaced with zeros");

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
      .num_lowercasings = absl::GetFlag(FLAGS_num_lowercasings),
      .num_address_replacements = absl::GetFlag(FLAGS_num_address_replacements),
      .num_phone_replacements = absl::GetFlag(FLAGS_num_phone_replacements),
      .num_context_drops_between_labels =
          absl::GetFlag(FLAGS_num_context_drops_between_labels),
      .num_context_drops_outside_one_label =
          absl::GetFlag(FLAGS_num_context_drops_outside_one_label),
      .probability_per_drop = absl::GetFlag(FLAGS_probability_per_drop),
      .mask_digits = absl::GetFlag(FLAGS_mask_digits)};

  for (const std::string& corpus : corpora) {
    augmenter::TextprotoIO textproto_io = augmenter::TextprotoIO();
    if (!textproto_io.Load(corpus)) {
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

    augmenter::Augmenter augmenter =
        augmenter::Augmenter(textproto_io.documents(), augmentations,
                             &address_sampler, &phones_sampler);
    augmenter.Augment();
    textproto_io.set_documents(augmenter.documents());
    textproto_io.Save(corpus);
  }

  return 0;
}

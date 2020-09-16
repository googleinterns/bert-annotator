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

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "augmenter/augmenter.h"
#include "augmenter/textproto_io.h"

ABSL_FLAG(double, lowercase, 0, "Percentage of augmentations by lowercasing");
ABSL_FLAG(std::vector<std::string>, corpora, std::vector<std::string>({}),
          "comma-separated list of corpora to augment");

// Augments the dataset by applying configurable actions, see defined flags.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  absl::ParseCommandLine(argc, argv);

  const double lowercase_percentage = absl::GetFlag(FLAGS_lowercase);
  const std::vector<std::string> corpora = absl::GetFlag(FLAGS_corpora);

  for (const std::string& corpus : corpora) {
    std::cout << corpus << std::endl;
    augmenter::TextprotoIO textproto_io = augmenter::TextprotoIO();
    if (!textproto_io.load(corpus)) {
      std::cerr << "Skipping corpus " << corpus << "." << std::endl;
      continue;
    }

    augmenter::Augmenter augmenter =
        augmenter::Augmenter(textproto_io.get_documents());
    augmenter.lowercase(lowercase_percentage);

    textproto_io.set_documents(augmenter.get_documents());
    textproto_io.save(corpus);
  }

  return 0;
}

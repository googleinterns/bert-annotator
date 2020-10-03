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

#ifndef AUGMENTER_AUGMENTATIONS_H_
#define AUGMENTER_AUGMENTATIONS_H_

namespace augmenter {

struct Augmentations {
  int num_total;
  int num_lowercasings_complete_token;
  double probability_per_lowercasing_complete_token;
  int num_lowercasings_first_letter;
  double probability_per_lowercasing_first_letter;
  int num_uppercasings_complete_token;
  double probability_per_uppercasing_complete_token;
  int num_uppercasings_first_letter;
  double probability_per_uppercasing_first_letter;
  int num_address_replacements;
  int num_phone_replacements;
  int num_context_drops_between_labels;
  int num_context_drops_outside_one_label;
  double probability_per_drop;
  int num_contextless_addresses;
  int num_contextless_phones;
  bool mask_digits;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTATIONS_H_

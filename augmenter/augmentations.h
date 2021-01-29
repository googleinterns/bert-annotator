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
  double prob_lowercasing_complete_token;
  double prob_lowercasing_first_letter;
  double prob_uppercasing_complete_token;
  double prob_uppercasing_first_letter;
  double prob_address_replacement;
  double prob_phone_replacement;
  double prob_context_drop_between_labels;
  double prob_context_drop_outside_one_label;
  double prob_punctuation_change_between_tokens;
  double prob_punctuation_change_at_sentence_end;
  double prob_sentence_concatenation;
  bool modify_original;
  int num_contextless_addresses;
  int num_contextless_phones;
  bool mask_digits;
  bool shuffle;
};

}  // namespace augmenter

#endif  // AUGMENTER_AUGMENTATIONS_H_

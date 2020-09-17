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

#ifndef AUGMENTER_RANDOM_SAMPLER_H_
#define AUGMENTER_RANDOM_SAMPLER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "augmenter/random_item.h"

namespace augmenter {

class RandomSampler {
 public:
  explicit RandomSampler(std::istringstream& input_stream);
  RandomSampler(std::istringstream& input_stream, absl::BitGenRef bitgen);
  const std::string Sample();
  std::vector<RandomItem> items();

 private:
  const std::string Search(const double accumulated_probability);
  const std::string Search(const double accumulated_probability,
                           const int lower_bound, const int upper_bound);
  std::vector<RandomItem> random_items_;
  absl::BitGenRef bitgenref_;
  absl::BitGen bitgen_;
};

}  // namespace augmenter

#endif  // AUGMENTER_RANDOM_SAMPLER_H_

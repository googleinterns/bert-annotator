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

#ifndef AUGMENTER_PERCENTAGE_H_
#define AUGMENTER_PERCENTAGE_H_

#include <string>

#include "absl/flags/flag.h"

namespace augmenter {

struct Percentage {
  explicit Percentage(int p = 0) : percentage(p) {}

  int percentage;  // Valid range is [0..100]
};
std::string AbslUnparseFlag(Percentage p);

bool AbslParseFlag(absl::string_view text, Percentage* p, std::string* error);

}  // namespace augmenter

#endif  // AUGMENTER_PERCENTAGE_H_

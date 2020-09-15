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

#include "augmenter/percentage.h"

#include <fcntl.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

namespace augmenter {

// Returns a textual flag value corresponding to the Percentage `p`.
std::string AbslUnparseFlag(Percentage p) {
  // Delegate to the usual unparsing for int.
  return absl::UnparseFlag(p.percentage);
}

// Parses a Percentage from the command line flag value `text`.
// Returns true and sets `*p` on success; returns false and sets `*error`
// on failure.
bool AbslParseFlag(absl::string_view text, Percentage* p, std::string* error) {
  // Convert from text to int using the int-flag parser.
  if (!absl::ParseFlag(text, &p->percentage, error)) {
    return false;
  }
  if (p->percentage < 0 || p->percentage > 100) {
    *error = "not in range [0,100]";
    return false;
  }
  return true;
}

}  // namespace augmenter

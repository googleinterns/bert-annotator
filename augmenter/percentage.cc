#include "augmenter/percentage.h"

#include <fcntl.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

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

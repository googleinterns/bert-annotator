#include "augmenter/main.h"

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "augmenter/augmenter.h"
#include "augmenter/percentage.h"
#include "augmenter/textproto_io.h"

ABSL_FLAG(Percentage, lowercase, Percentage(0),
          "Percentage of augmentations by lowercasing");
ABSL_FLAG(std::vector<std::string>, corpora, std::vector<std::string>({}),
          "comma-separated list of corpora to augment");

// Main function:  Reads the entire address book from a file and prints all
//   the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  absl::ParseCommandLine(argc, argv);

  Percentage lowercase_percentage = absl::GetFlag(FLAGS_lowercase);
  std::vector<std::string> corpora = absl::GetFlag(FLAGS_corpora);

  for (std::string corpus : corpora) {
    std::cout << corpus << std::endl;
    TextprotoIO textproto_io = TextprotoIO();
    if (!textproto_io.load(corpus)) {
      std::cerr << "Skipping corpus " << corpus << "." << std::endl;
      continue;
    }

    Augmenter augmenter = Augmenter(textproto_io.get_documents());
    augmenter.lowercase(lowercase_percentage);

    textproto_io.set_documents(augmenter.get_documents());
    textproto_io.save(corpus);
  }

  return 0;
}
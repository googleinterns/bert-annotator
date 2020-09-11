#include "augmenter/textproto_io.h"


#include <fstream>
#include <string>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "protocol_buffer/documents.pb.h"

bool TextprotoIO::load(std::string corpus) {
  std::ifstream input("data/input/preprocessed/" + corpus + ".textproto");
  if (input.fail()) {
    std::cerr << "Failed to load corpus " << corpus << std::endl;
    return false;
  }
  google::protobuf::io::IstreamInputStream fileInput(&input, std::ios::binary);
  if (!google::protobuf::TextFormat::Parse(&fileInput, &documents_)) {
    std::cerr << "Failed to parse corpus " << corpus << std::endl;
    return false;
  }

  return true;
}

bool TextprotoIO::save(std::string corpus) {
  std::ofstream output("data/output/" + corpus + ".textproto");
  google::protobuf::io::OstreamOutputStream fileOutput(&output,
                                                       std::ios::binary);
  if (!google::protobuf::TextFormat::Print(documents_, &fileOutput)) {
    std::cerr << "Failed to save document." << std::endl;
    return false;
  }

  return true;
}

bert_annotator::Documents TextprotoIO::get_documents() { return documents_; }

void TextprotoIO::set_documents(bert_annotator::Documents documents) {
  documents_ = documents;
}
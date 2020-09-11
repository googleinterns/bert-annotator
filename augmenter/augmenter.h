#ifndef BERTANNOTATOR_AUGMENTER_AUGMENTER_H_
#define BERTANNOTATOR_AUGMENTER_AUGMENTER_H_
#include "augmenter/percentage.h"
#include "protocol_buffer/documents.pb.h"

class Augmenter {
 public:
  Augmenter(bert_annotator::Documents documents);
  void lowercase(Percentage lowercase_percentage);
  bert_annotator::Documents get_documents();

 private:
  bert_annotator::Documents documents_;
};
#endif  // BERTANNOTATOR_AUGMENTER_AUGMENTER_H_